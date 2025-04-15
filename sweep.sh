# given a folder of yaml files, run the sweep
# folder should be required as an argument

FOLDER=$1

for file in $FOLDER/*.yaml; do
    filename=$(basename "$file" .yaml)
    # skip if filename is template.yaml
    if [[ "$filename" == "template" ]]; then
        continue
    fi
    nlprun -q jag -g 1 -a boundless -n "$filename" "uv run tinylang '$file' --wandb"
done

# nlprun -q jag -g 1 -a boundless -n 4_4_128_llama 'uv run tinylang experiments/configs/pcfg_4_4_128_llama.yaml'
# nlprun -q jag -g 1 -a boundless -n 4_4_64_llama 'uv run tinylang experiments/configs/pcfg_4_4_64_llama.yaml'
# nlprun -q jag -g 1 -a boundless -n 4_4_32_llama 'uv run tinylang experiments/configs/pcfg_4_4_32_llama.yaml'
# nlprun -q jag -g 1 -a boundless -n 4_4_16_llama 'uv run tinylang experiments/configs/pcfg_4_4_16_llama.yaml'

# nlprun -q jag -g 1 -a boundless -n 3_2_256_right 'uv run tinylang experiments/configs/pcfg_3_2_256_right.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_128_right 'uv run tinylang experiments/configs/pcfg_3_2_128_right.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_64_right 'uv run tinylang experiments/configs/pcfg_3_2_64_right.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_16_right 'uv run tinylang experiments/configs/pcfg_3_2_16_right.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_4_right 'uv run tinylang experiments/configs/pcfg_3_2_4_right.yaml'

# nlprun -q jag -g 1 -a boundless -n 3_2_256 'uv run tinylang experiments/configs/pcfg_3_2_256.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_128 'uv run tinylang experiments/configs/pcfg_3_2_128.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_64 'uv run tinylang experiments/configs/pcfg_3_2_64.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_16 'uv run tinylang experiments/configs/pcfg_3_2_16.yaml'
# nlprun -q jag -g 1 -a boundless -n 3_2_4 'uv run tinylang experiments/configs/pcfg_3_2_4.yaml'

# nlprun -q jag -g 1 -a boundless -n 2_2_16 'uv run tinylang experiments/configs/pcfg_2_2_16.yaml'
# nlprun -q jag -g 1 -a boundless -n 2_2_64 'uv run tinylang experiments/configs/pcfg_2_2_64.yaml'
# nlprun -q jag -g 1 -a boundless -n 2_2_128 'uv run tinylang experiments/configs/pcfg_2_2_128.yaml'
# nlprun -q jag -g 1 -a boundless -n 2_2_256 'uv run tinylang experiments/configs/pcfg_2_2_256.yaml'

