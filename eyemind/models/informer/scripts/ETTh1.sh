### M

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 48 --label_length 48 --pred_length 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 96 --label_length 48 --pred_length 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 168 --label_length 168 --pred_length 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 168 --label_length 168 --pred_length 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTh1 --features M --seq_len 336 --label_length 336 --pred_length 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

### S

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_length 168 --pred_length 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_length 168 --pred_length 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_length 336 --pred_length 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_length 336 --pred_length 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTh1 --features S --seq_len 720 --label_length 336 --pred_length 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5