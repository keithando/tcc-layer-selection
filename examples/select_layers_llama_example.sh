python3 select_layers_decoder.py --model_name meta-llama/Llama-2-7b-hf  \
                     --task python \
                     --samples 128  \
                     --seed 42 \
                     --ratio_of_layers 0.5 \
                     --output_dir ./llama_layer_selection_output \
                     --batch_size 8 \