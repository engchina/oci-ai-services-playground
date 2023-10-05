# oci-ai-services-playground

(**Unofficial**) OCI AI Services Playground

## prepare

create conda environment,

```
conda create -n oci-ai-services-playground python=3.10 -y
conda activate oci-ai-services-playground
```

install requirements,

```
pip install -r requirements.txt
```

create .env file,

```
cp .env.example .env
```

modify .env file,

```
vi .env

---
COMPARTMENT="Specify your compartmentId here"
---
```

# launch language playground

```
python language.py
```


## access language playground

open [http://127.0.0.1:7860](http://127.0.0.1:7860) or [http://127.0.0.1:7860/?__theme=dark](http://127.0.0.1:7860/?__theme=dark)
