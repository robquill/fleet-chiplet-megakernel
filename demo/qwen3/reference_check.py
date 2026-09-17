"""Compare mirage's Qwen3-30B-A3B megakernel output against a plain
torch/transformers forward pass, to localize correctness bugs.

Used during the num-layers=3+ wrong-output investigation (see the
TRACE_VALUES diagnostics commit) to bisect which layer count first
diverges from a real transformers forward pass, and to compare
intermediate per-layer activations.

Modes (pick one):
  --generate            Greedily generate --max-new-tokens tokens and print
                         their ids/text. Compare against demo_30B_A3B.py's
                         printed `new_token_ids` for the same --num-layers,
                         --prompt and prefix of the generated sequence.
  --bisect N1,N2,...     For each layer count, run a single forward pass and
                         print the argmax next-token id. Loads the model once
                         and truncates model.model.layers per layer count, so
                         it's much faster than --generate for bisecting which
                         layer count first diverges from mirage's output.
  --trace-activations    Register forward hooks on every decoder layer
                         (input_layernorm, self_attn, post_attention_layernorm
                         in/out, layer out) plus embed_tokens/norm, and print
                         a cheap checksum (sum of the last token's first 8
                         hidden dims) for each. Compare against mirage's
                         TRACE_VALUE/TRACE_W13_WRITE/TRACE_SILU_IN/etc prints
                         (see persistent_kernel.cuh) for the same op.

--num-layers truncates model.model.layers to the first N decoder layers,
mirroring what demo_30B_A3B.py's --num-layers does in the megakernel task
graph -- pass the same value to both to compare apples to apples.
"""
import argparse
import torch
from transformers import Qwen3MoeForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "Give me a short introduction to large language model."


def load_model(model_name: str, num_layers: int | None):
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = Qwen3MoeForCausalLM.from_pretrained(model_name).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if num_layers is not None:
        model.model.layers = torch.nn.ModuleList(list(model.model.layers)[:num_layers])
        model.config.num_hidden_layers = num_layers
    return model, tokenizer


def build_inputs(model, tokenizer, prompt: str):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    return model_inputs


def sum8(t: torch.Tensor) -> float:
    """Checksum of the last token position's first 8 hidden dims."""
    return float(t.reshape(-1, t.shape[-1])[-1, :8].float().sum())


def run_generate(args):
    model, tokenizer = load_model(args.model, args.num_layers)
    model_inputs = build_inputs(model, tokenizer, args.prompt)
    prompt_len = model_inputs.input_ids.shape[-1]
    print(f"prompt_len={prompt_len}")

    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    new_tokens = generated[0, prompt_len:]
    print("reference token ids:", new_tokens.tolist())
    print("reference text:", repr(tokenizer.decode(new_tokens, skip_special_tokens=True)))


def run_bisect(args):
    layer_counts = [int(x) for x in args.bisect.split(",")]
    model, tokenizer = load_model(args.model, num_layers=None)
    full_layers = list(model.model.layers)
    model_inputs = build_inputs(model, tokenizer, args.prompt)
    print(f"prompt_len={model_inputs.input_ids.shape[-1]}")

    for n in layer_counts:
        model.model.layers = torch.nn.ModuleList(full_layers[:n])
        model.config.num_hidden_layers = n
        with torch.no_grad():
            out = model(**model_inputs)
        next_id = int(out.logits[0, -1].argmax())
        print(f"num_layers={n} first_token_id={next_id} token_text={tokenizer.decode([next_id])!r}")


def run_trace_activations(args):
    model, tokenizer = load_model(args.model, args.num_layers)
    model_inputs = build_inputs(model, tokenizer, args.prompt)
    print(f"prompt_len={model_inputs.input_ids.shape[-1]}")

    records = []

    def mk_pre_hook(label):
        def hook(module, hook_args, kwargs):
            x = hook_args[0] if hook_args else kwargs["hidden_states"]
            records.append((label + "_input", sum8(x)))
        return hook

    def mk_post_hook(label):
        def hook(module, hook_args, output):
            out = output[0] if isinstance(output, tuple) else output
            records.append((label + "_output", sum8(out)))
        return hook

    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm.register_forward_hook(mk_post_hook(f"L{i}_input_layernorm"))
        layer.self_attn.register_forward_hook(mk_post_hook(f"L{i}_attn"))
        layer.post_attention_layernorm.register_forward_pre_hook(mk_pre_hook(f"L{i}_post_attn_ln"), with_kwargs=True)
        layer.post_attention_layernorm.register_forward_hook(mk_post_hook(f"L{i}_post_attn_ln"))
        layer.register_forward_hook(mk_post_hook(f"L{i}_layer"))

    model.model.embed_tokens.register_forward_hook(mk_post_hook("embed"))
    model.model.norm.register_forward_hook(mk_post_hook("final_norm"))

    with torch.no_grad():
        out = model(**model_inputs)

    for label, val in records:
        print(f"{label}: sum8={val:.6f}")

    next_id = int(out.logits[0, -1].argmax())
    print("next_token_id:", next_id, repr(tokenizer.decode([next_id])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true", help="Greedily generate tokens")
    mode.add_argument("--bisect", type=str, metavar="N1,N2,...", help="Comma-separated layer counts to compare first-token predictions across")
    mode.add_argument("--trace-activations", action="store_true", help="Print per-layer activation checksums")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--num-layers", type=int, default=None, help="Truncate to first N decoder layers (--generate/--trace-activations only; default: all)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8, help="--generate only")
    args = parser.parse_args()

    if args.generate:
        run_generate(args)
    elif args.bisect:
        run_bisect(args)
    else:
        run_trace_activations(args)
