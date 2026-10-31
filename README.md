# Team Algo Alphas

## How to Rerun the Research

From the project root directory, activate the virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the research pipeline:

```bash
python main.py
```

To verify existing outputs without running models:

```bash
python main.py --verify
```

---

## How to Open the Streamlit Dashboard


```bash
python -m streamlit run app.py
```
