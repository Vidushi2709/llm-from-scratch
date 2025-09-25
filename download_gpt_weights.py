'''import urllib.request

url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch05/"
    "01_main-chapter-code/gpt_download.py"
)

filename = url.split("/")[-1]  # this gives "gpt_download.py"
urllib.request.urlretrieve(url, filename)

print(f"Downloaded file saved as: {filename}")'''

from gpt_download import download_and_load_gpt2
settings, params= download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print(settings)
print(params)