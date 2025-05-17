## Créer un nouvel outil LangChain

### 1. Choisir une catégorie
Organisez vos outils dans `app/tools/<catégorie>/`, par exemple `media`, `web`, `data`, `communication`, etc.

### 2. Créer l’arborescence
```bash
mkdir -p app/tools/<catégorie>/<outil_name>
cd app/tools/<catégorie>/<outil_name>

```

### 3. Fichiers à générer

- `schema.py` : definitions Pydantic
- `core.py` : logique métier pure
- `tool.py` : wrapper et enregistrement
- `README.md` : installation, usage, variables d’environ
- `requirements.txt` : dépendances spécifiques
- `.env.example` : variables d’environnement à renseigner

### 4. schema.py

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class <ToolName>Input(BaseModel):
    """Schéma des paramètres"""
    param1: str = Field(..., description="Description de param1")
    param2: Optional[int] = Field(None, description="Description de param2")

class <ToolName>Output(BaseModel):
    """Schéma de la réponse"""
    result: List[Dict[str, Any]]

```

### 5. core.py

```python
import requests

def run_<outil_name>(param1: str, param2: Optional[int] = None) -> List[Dict]:
    """Logique appel API externe"""
    resp = requests.get(
        "https://api.example.com/endpoint",
        params={"param1": param1, "param2": param2}
    )
    resp.raise_for_status()
    return resp.json().get("data", [])

```

### 6. tool.py

```python
from app.tools.registry import register
from langchain.tools import StructuredTool
from app.tools.<catégorie>.<outil_name>.schema import <ToolName>Input, <ToolName>Output
from app.tools.<catégorie>.<outil_name>.core import run_<outil_name>

@register(
    name="<outil_name>",
    description="Description courte de l’outil",
    args_schema=<ToolName>Input
)
def <outil_name>(param1: str, param2: Optional[int] = None) -> <ToolName>Output:
    data = run_<outil_name>(param1, param2)
    return <ToolName>Output(result=data)

```

### 7. README.md

```markdown
# Outil `<outil_name>`

## Installation
```bash
pip install -r requirements.txt

```

## Configuration

Copiez `.env.example` → `.env` et renseignez vos clés :

```
API_KEY=...
ENDPOINT_URL=...

```

## Usage

```python
from app.tools.registry import load_tools

tools = load_tools(["<outil_name>"])
agent = OpenAIAgent(..., tools=tools)

response = agent.run({
    "input": {"param1": "foo", "param2": 42}
})
print(response)

```

```

### 8. tests/unit/<outil_name>_test.py
```python
from app.tools.<catégorie>.<outil_name>.tool import <outil_name>

def test_<outil_name>():
    out = <outil_name>(param1="foo")
    assert hasattr(out, 'result')

```

```

---

## Bonnes pratiques

- **Logging** : utilisez `get_logger(__name__)` et les niveaux appropriés.
- **Retry** : décorateur `@with_retry(max_retries=3, delay=1)` pour appels réseau exposés.
- **Validation** : vérifiez les clés API à l’initialisation (settings).
- **Docs** : maintenez `README.md` et `.env.example` à jour.
- **CI** : ajoutez un job de test unitaire et un linting (flake8, mypy).

---

Avec ces instructions, votre équipe pourra refactoriser les outils existants et créer de nouveaux outils de manière standardisée et rapide !