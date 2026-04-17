# NAISC2026-wasurenai

## inference server 
ignore ```inference_server```, the only possibly important part of it is the meralion v2 extraction in it, i used the csv file from running that code (but i did not preserve the sussy structure of splitting time and day and whatever) 


## Web App

The repository root now includes a small Flask app that serves the site in
`static_site/`.

Run it from the repository root with:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

