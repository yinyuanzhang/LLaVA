from llava.train.train import train
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    train(attn_implementation="flash_attention_2")
