# Agent IA Conversationnel avec LangChain

Une API REST pour un agent conversationnel IA basé sur LangChain, facilement intégrable dans vos applications.

## Fonctionnalités principales

- 💬 Conversation avec un agent IA via LangChain
- 🧠 Gestion de la mémoire des conversations
- 🔑 Authentification par clé API
- 🛠️ Sessions configurables (modèle, température, etc.)
- 📊 Logging complet des interactions

> ***LE DISCORD 👉🏻 https://discord.gg/T6DCneUhD7***

> ***TOUS LES OUTILS CONÇUS POUR L'AGENT 👉🏻 https://github.com/Chugyy/agent-tools***

## Guide de démarrage rapide

### Prérequis

- Python 3.9 ou supérieur
- Une clé API OpenAI

### Installation

#### 1. Cloner le dépôt

```bash
git clone https://github.com/Chugyy/langchain-ai-agentv2.git
cd langchain-ai-agentv2
```

#### 2. Créer un environnement virtuel

```bash
# Sur macOS/Linux
python -m venv venv
source venv/bin/activate

# Sur Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### 4. Configurer les variables d'environnement

Créez un fichier `.env` à la racine du projet avec le contenu suivant:

```
# Configuration du LLM
OPENAI_API_KEY=sk-votre-clé-api-openai
LLM_NAME=gpt-gpt-4-0613
TEMPERATURE=0.7

# Outils activés
ENABLED_TOOLS=shout,file_loader

# Configuration du serveur
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

> **Important**: Remplacez `sk-votre-clé-api-openai` par votre véritable clé API OpenAI.

#### 5. Démarrer le serveur

```bash
# Mode développement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 6. Tester l'API

Voici quelques exemples pour tester les différents endpoints de l'API :

```bash
# Message simple
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer admin-key-for-development" \
  -d '{
    "message": "Bonjour, comment vas-tu?"
  }'

# Message avec média
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer admin-key-for-development" \
  -d '{
    "message": "Analyse cette image",
    "media": [
      {
        "url": "https://example.com/image.jpg",
        "reference_id": "image1",
        "title": "Mon image",
        "description": "Description optionnelle"
      }
    ]
  }'

# Récupérer l'historique d'une session
curl -X GET http://localhost:8000/sessions/votre-session-id \
  -H "Authorization: Bearer admin-key-for-development"

# Mettre à jour la configuration d'une session
curl -X PATCH http://localhost:8000/sessions/votre-session-id/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer admin-key-for-development" \
  -d '{
    "temperature": 0.8,
    "model_name": "gpt-4"
  }'

# Générer une nouvelle clé API (admin seulement)
curl -X POST http://localhost:8000/auth/keys \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer admin-key-for-development" \
  -d '{
    "scopes": ["chat"],
    "expires_in_days": 30,
    "rate_limit": 100
  }'

# Vérifier l'état de l'API
curl -X GET http://localhost:8000/health
```

## Documentation de l'API

Les principaux endpoints sont:

- `POST /chat` - Envoyer un message à l'agent
- `GET /sessions/{session_id}` - Récupérer l'historique d'une session
- `PATCH /sessions/{session_id}/config` - Mettre à jour la configuration d'une session
- `POST /auth/keys` - Générer une nouvelle clé API (admin seulement)
- `GET /health` - Vérifier que l'API fonctionne

Pour une documentation détaillée, consultez `docs/Context.md` ou l'interface Swagger UI (`/docs`).

## Configuration

La configuration se fait via les variables d'environnement ou le fichier `.env`:

- `OPENAI_API_KEY` - Votre clé API OpenAI
- `LLM_NAME` - Le modèle à utiliser (ex: "gpt-4o-mini")
- `TEMPERATURE` - Réglage de température pour la génération (0.0-1.0)
- `ENABLED_TOOLS` - Liste des outils activés
- `ADMIN_API_KEY` - Clé admin pour les opérations privilégiées

## Outils disponibles

Les outils de base inclus sont:

- **date** - Permet au LLM de caculer des dates
- **media** - Charge tous les contenus possible envoyé au LLM en texte

## Structure du projet

```
app/
├── agents/          # Implémentation de l'agent
├── api/             # Endpoints API
├── llm/             # Factory LLM 
├── memory/          # Gestion de la mémoire
├── tools/           # Outils de l'agent
└── utils/           # Utilitaires (settings, logging, etc.)
```

## Tests

Pour exécuter les tests:

```bash
pytest
```