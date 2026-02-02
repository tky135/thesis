import contextlib
import fire
import mup
import numpy as np
import lib.datasets
import lib.models
import lib.utils
import os
import time
import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim, autograd


# ============================================================================
# Perplexity Evaluation Function
# ============================================================================

def compute_perplexity(
    texts,
    model=None,
    tokenizer=None,
    model_name="gpt2",
    device="cuda",
    max_length=1024,
    stride=512,
):
    """
    Compute perplexity of generated texts using a standard language model.
    
    Args:
        texts: List of strings to evaluate.
        model: Pre-loaded model (optional, will load if None).
        tokenizer: Pre-loaded tokenizer (optional, will load if None).
        model_name: HuggingFace model name if model/tokenizer not provided.
        device: Device to use.
        max_length: Maximum sequence length for the evaluation model.
        stride: Stride for sliding window on long sequences.
    
    Returns:
        dict with per-text perplexities, mean perplexity, and losses.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_losses = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True,
            )
            
            seq_len = encodings.input_ids.size(1)
            
            if seq_len <= max_length:
                input_ids = encodings.input_ids.to(device)
                target_ids = input_ids.clone()
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss.item()
            else:
                # Sliding window for long sequences
                nlls = []
                prev_end_loc = 0
                
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - prev_end_loc
                    
                    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                    target_ids = input_ids.clone()
                    target_ids[:, :-trg_len] = -100
                    
                    outputs = model(input_ids, labels=target_ids)
                    nlls.append(outputs.loss.item() * trg_len)
                    
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break
                
                neg_log_likelihood = sum(nlls) / prev_end_loc
            
            all_losses.append(neg_log_likelihood)
    
    perplexities = [np.exp(loss) for loss in all_losses]
    mean_perplexity = np.exp(np.mean(all_losses))
    
    return {
        "perplexity": perplexities,
        "mean_perplexity": mean_perplexity,
        "loss": all_losses,
    }


def main(**args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = lib.utils.AttributeDict(args)
    args.setdefault('seq_len', 256)
    args.setdefault('vocab_size', 128001)
    args.setdefault('weights_path', None)
    args.setdefault('dim', 384)
    args.setdefault('n_blocks', 3)
    args.setdefault('n_heads', 6)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('initial_noise_scale', 1.0)
    args.setdefault('n_samples', 32)
    args.setdefault('sampling_timesteps', 4096)
    args.setdefault('score_temp', 0.9)
    args.setdefault('output_scale', 1.)
    args.setdefault('owt2_tokenizer', True)
    args.setdefault('ddim_sampler', False)
    args.setdefault('guidance_weight', 2.)
    # New args for perplexity evaluation
    args.setdefault('ppl_model', 'gpt2')
    args.setdefault('eval_perplexity', True)

    lib.utils.print_args(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_device('cuda')

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    # ========================================================================
    # Load perplexity evaluation model once (to avoid repeated loading)
    # ========================================================================
    ppl_model = None
    ppl_tokenizer = None
    if args.eval_perplexity:
        print(f"\nLoading perplexity evaluation model: {args.ppl_model}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)
        ppl_model = AutoModelForCausalLM.from_pretrained(args.ppl_model)
        ppl_model.to('cuda')
        ppl_model.eval()
        if ppl_tokenizer.pad_token is None:
            ppl_tokenizer.pad_token = ppl_tokenizer.eos_token
        print(f"Perplexity model loaded successfully.\n")
        
    def log1mexp(x):
        # Computes log(1-exp(-|x|))
        x = -x.abs()
        return torch.where(
            x > -0.693,
            torch.log(-torch.expm1(x)),
            torch.log1p(-torch.exp(x))
        )

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(args.vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, args.vocab_size).float()
        }
        
    def evaluate_and_print_ppl(texts, task_name):
        """Helper function to evaluate and print perplexity results."""
        if not args.eval_perplexity:
            return
        print(f"\n{'='*60}")
        print(f"Perplexity Evaluation ({args.ppl_model}) - {task_name}")
        print(f"{'='*60}")
        results = compute_perplexity(
            texts, 
            model=ppl_model, 
            tokenizer=ppl_tokenizer,
            device='cuda'
        )
        for i, (text, ppl) in enumerate(zip(texts, results["perplexity"])):
            # Truncate text for display
            display_text = text[:80] + "..." if len(text) > 80 else text
            display_text = display_text.replace("\n", "↵")
            print(f"  Sample {i+1}: PPL = {ppl:.2f}")
        print(f"  Mean Perplexity: {results['mean_perplexity']:.2f}")
        print(f"{'='*60}\n")
        return results
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()

    print(f'Loading weights from {args.weights_path}')
    for name, module in modules.items():
        module.load_state_dict(torch.load(
            os.path.join(args.weights_path, f'{name}.pt'),
            map_location=torch.device('cuda')
        ))

    for key in modules:
        print(key+':')
        lib.utils.print_model(modules[key])


    def generate_samples(guidance_tokens, seq_len=args.seq_len):
        """
        Sampling (implements Appendix A.4 eqn 33 in VDM). Needs float64 to work.
        guidance_tokens: [(token, weight, position, complement), ...]
            token: vocab index of token
            weight: guidance weight
            position: sequence index, or 'any', or 'all'
            complement: if True, do guidance on log(1-p(y|x))
        """
        with torch.no_grad():
            embedding_matrix = modules['embedding_matrix']()

            gamma_0, gamma_1 = modules['gamma_bounds']()
            alpha_0 = torch.sigmoid(-gamma_0).sqrt()
            sigma_0 = torch.sigmoid(gamma_0).sqrt()

            z = torch.randn((args.n_samples, seq_len, args.embed_dim), device='cuda') * args.initial_noise_scale
            x_selfcond = torch.zeros_like(z).float()
            for i, t in enumerate(tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps))):
                t = t[None].cuda()
                s = t - 1. / args.sampling_timesteps
                gamma_s = modules['noise_schedule'](s).double()
                gamma_t = modules['noise_schedule'](t).double()
                gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
                gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
                alpha_squared_s = torch.sigmoid(-gamma_s)
                alpha_squared_t = torch.sigmoid(-gamma_t)
                alpha_s = alpha_squared_s.sqrt()
                alpha_t = alpha_squared_t.sqrt()
                sigma_squared_s = torch.sigmoid(gamma_s)
                sigma_squared_t = torch.sigmoid(gamma_t)
                sigma_s = sigma_squared_s.sqrt()
                sigma_t = sigma_squared_t.sqrt()

                if len(guidance_tokens) > 0:
                    with torch.enable_grad():
                        z.requires_grad = True
                        logits, x_reconst = modules['model'](
                            z=z.to(torch.float32, copy=True),
                            gamma=gamma_t.float(),
                            embedding_matrix=embedding_matrix,
                            bias_scale=1.,
                            x_selfcond=x_selfcond
                        )

                        logprobs = F.log_softmax(logits.float(), dim=2)
                        logprobs_any = logprobs.logsumexp(dim=1)-float(seq_len)

                        sum_logp = 0.
                        for token, weight, position, complement in guidance_tokens:
                            if position == 'any':
                                logp = logprobs_any[:, token]
                            elif position == 'all':
                                logp = logprobs[:, :, token]
                            else:
                                logp = logprobs[:, position, token]
                            if complement:
                                logp = log1mexp(logp)
                            sum_logp += weight * logp.sum()

                        guidance_grad = autograd.grad(sum_logp, [z])[0]
                        z.requires_grad = False
                    x_selfcond = x_reconst.clone().detach()
                    x_reconst = x_reconst.double()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                    epsilon_pred /= args.score_temp
                    x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                    x_reconst += guidance_grad.double() * sigma_squared_t / alpha_squared_t.sqrt()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                else:
                    _, x_reconst = modules['model'](
                        z=z.to(torch.float32, copy=True),
                        gamma=gamma_t.float(),
                        embedding_matrix=embedding_matrix,
                        bias_scale=1.,
                        x_selfcond=x_selfcond
                    )
                    x_selfcond = x_reconst.clone().detach()
                    x_reconst = x_reconst.double()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                    epsilon_pred /= args.score_temp
                    x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                if t > 0:
                    if args.ddim_sampler:
                        z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
                    else:
                        c = -torch.expm1(gamma_s - gamma_t)
                        z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                        z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                        z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)

            logits, _ = modules['model'](
                z=z.float(),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond
            )
            x_samples = logits.argmax(dim=-1)

            return x_samples

    def print_samples(x_samples):
        if args.owt2_tokenizer:
            ret_list = []
            owt2_tokenizer = lib.datasets.deberta_tokenizer()
            for x in x_samples:
                x = owt2_tokenizer.decode(x.tolist(), skip_special_tokens=False)
                print(x.replace("\n", "↵"))
                ret_list.append(x.replace("\n", "↵"))
            return ret_list
        else:
            for x in x_samples:
                x = x.tolist()
                x = [idx2word[i].decode('utf-8', 'ignore') for i in x]
                x = ' '.join(x)
                x = x.replace('START','')
                x = x.replace('END','')
                x = x.replace('PAD','')
                x = x.replace(' .', '.')
                x = x.replace(' !', '!')
                x = x.replace(' ,', ',')
                x = x.replace(' \' ', '\'')
                x = x.strip()
                # replace newlines with '↵' symbol for cleaner printing
                print(x.replace("\n", "↵"))

    tokenizer = lib.datasets.deberta_tokenizer()

    # ========================================================================
    # 1. Unconditional Generation
    # ========================================================================
    print('Unconditional:')
    x_samples = generate_samples([], seq_len=256)
    texts = print_samples(x_samples)
    evaluate_and_print_ppl(texts, "Unconditional Generation")
    print("\n"*10)

    # ========================================================================
    # 2. Prefix Completion
    # ========================================================================
    prefixes = [
        ' This easy chicken curry recipe is made with just a handful of ingredients',
        ' Generative models of text are very versatile: they can be used'
    ]
    for prefix in prefixes:
        print('Prefix completion: ', prefix)
        prefix_tokens = tokenizer.encode(prefix)
        x_samples = generate_samples(
            [(token, args.guidance_weight, position, False) for position, token in enumerate(prefix_tokens)], seq_len=256
        )
        texts = print_samples(x_samples)
        evaluate_and_print_ppl(texts, f"Prefix Completion: '{prefix[:50]}...'")
        print("\n"*10)

    # ========================================================================
    # 3. Infilling
    # ========================================================================
    print('Infilling: A year ago in Paris, [...] Wow, what a great day!')
    tokenizer = lib.datasets.deberta_tokenizer()
    prefix = tokenizer.encode(' A year ago in Paris,')
    suffix = tokenizer.encode('. Wow, what a great day!')
    infill_len = 40
    x_samples = generate_samples(
        [(token, args.guidance_weight, position, False) for position, token in enumerate(prefix)]
        + [(token, args.guidance_weight, position + len(prefix) + infill_len, False) for position, token in enumerate(suffix)], seq_len=256
    )
    texts = print_samples(x_samples)
    evaluate_and_print_ppl(texts, "Infilling")
    print("\n"*10)

    # # ========================================================================
    # # 5. Word-level weights: law[1] and medicine[10]
    # # ========================================================================
    # print('Word-level weights: Let\'s talk about law[1] and medicine[10].')
    # guidance = [
    #     (tokenizer.encode(' Let'),      args.guidance_weight,   0,  False),
    #     (tokenizer.encode('\'s'),       args.guidance_weight,   1,  False),
    #     (tokenizer.encode(' talk'),     args.guidance_weight,   2,  False),
    #     (tokenizer.encode(' about'),    args.guidance_weight,   3,  False),
    #     (tokenizer.encode(' law'),      args.guidance_weight,   4,  False),
    #     (tokenizer.encode(' and'),      args.guidance_weight,   5,  False),
    #     (tokenizer.encode(' medicine'), 10.,                    6,  False),
    #     (tokenizer.encode('.'),         args.guidance_weight,   7,  False),
    # ]
    # assert(all(len(a) == 1 for a,_,_,_ in guidance))
    # guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    # x_samples = generate_samples(guidance, seq_len=256)
    # texts = print_samples(x_samples)
    # evaluate_and_print_ppl(texts, "Word-level weights: law[1], medicine[10]")
    # print('\n'*10)

    # # ========================================================================
    # # 6. Lexically constrained generation: Donald
    # # ========================================================================
    # print(f'Lexically constrained generation: Donald')
    # guidance = [
    #     (tokenizer.encode(' Donald'), 3., 'any', False),
    # ]
    # assert(all(len(a) == 1 for a,_,_,_ in guidance))
    # guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    # x_samples = generate_samples(guidance, seq_len=256)
    # texts = print_samples(x_samples)
    # evaluate_and_print_ppl(texts, "Lexically Constrained: 'Donald'")
    # print("\n"*10)

    # # ========================================================================
    # # 7. Negation: Donald but not Trump
    # # ========================================================================
    # print(f'Negation: Donald but not Trump')
    # guidance = [
    #     (tokenizer.encode(' Donald'), 3., 'any', False),
    #     (tokenizer.encode(' Trump'), 10., 'all', True),
    # ]
    # assert(all(len(a) == 1 for a,_,_,_ in guidance))
    # guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    # x_samples = generate_samples(guidance, seq_len=256)
    # texts = print_samples(x_samples)
    # evaluate_and_print_ppl(texts, "Negation: 'Donald' but not 'Trump'")
    # print("\n"*10)

    # ========================================================================
    # Print summary of all perplexity results
    # ========================================================================
    if args.eval_perplexity:
        print("\n" + "="*70)
        print("PERPLEXITY EVALUATION COMPLETE")
        print(f"Evaluation model: {args.ppl_model}")
        print("="*70)


if __name__ == '__main__':
    fire.Fire(main)