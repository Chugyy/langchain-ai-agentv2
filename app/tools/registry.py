"""
Registre d'outils pour l'agent IA.
Permet d'enregistrer et charger dynamiquement des outils.
"""
from typing import Dict, List, Callable, Optional, Any, Type
import inspect
import importlib
import os
from functools import wraps
from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model
from app.utils.logging import get_logger

# Logger pour ce module
logger = get_logger(__name__)

# Registre global des outils
_TOOLS_REGISTRY: Dict[str, BaseTool] = {}

# Mapping des schémas pour chaque outil
_SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register(name: Optional[str] = None,
             description: Optional[str] = None,
             args_schema: Optional[Type[BaseModel]] = None):
    """
    Décorateur pour enregistrer une fonction comme outil.
    """
    def decorator(func):
        tool_name = name or func.__name__
        tool_desc = description or inspect.getdoc(func) or "No description"
        schema = args_schema or _build_schema(func)

        tool = StructuredTool.from_function(
            func=func,
            name=tool_name,
            description=tool_desc,
            args_schema=schema
        )

        _TOOLS_REGISTRY[tool_name] = tool
        _SCHEMA_REGISTRY[tool_name] = schema
        logger.info(f"Registered tool: {tool_name}")
        return func

    return decorator


def _build_schema(func) -> Type[BaseModel]:
    sig = inspect.signature(func)
    fields = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        annotation = param.annotation if param.annotation != inspect._empty else str
        default = param.default if param.default != inspect._empty else ...
        fields[name] = (annotation, default)
    schema_name = f"{func.__name__.capitalize()}Schema"
    return create_model(schema_name, **fields)


def _recursive_import_tools(path: str, package: str):
    """
    Importe dynamiquement tous les modules .py sous 'path', y compris sous-répertoires.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py') and file not in ('__init__.py', 'registry.py'):
                rel_dir = os.path.relpath(root, path)
                if rel_dir == '.':
                    module = f"{package}.{file[:-3]}"
                else:
                    subpkg = rel_dir.replace(os.sep, '.')
                    module = f"{package}.{subpkg}.{file[:-3]}"
                try:
                    importlib.import_module(module)
                    logger.debug(f"Imported module: {module}")
                except Exception as e:
                    logger.error(f"Failed to import {module}: {e}")


def load_all_tools() -> List[BaseTool]:
    """
    Charge tous les outils enregistrés dynamiquement.
    """
    tools_dir = os.path.dirname(__file__)
    # Adjust package name to correctly discover tools from app.tools.<category>.<module>
    # Assuming registry.py is in app/tools/
    # The package for tools will be app.tools
    package_parts = __name__.split('.') 
    if package_parts[-1] == 'registry': # if __name__ is app.tools.registry
        package = '.'.join(package_parts[:-1])
    else: # if __name__ is something else, perhaps during tests or direct execution
        # This might need adjustment based on actual execution context
        # For now, assume it's run in a way that __name__ reflects its path from 'app'
        package = 'app.tools'


    _recursive_import_tools(tools_dir, package)
    logger.info(f"Tools loaded: {list(_TOOLS_REGISTRY.keys())}")
    return list(_TOOLS_REGISTRY.values())


def load_tools(names: List[str]) -> List[BaseTool]:
    """
    Charge une liste d'outils par nom.
    """
    if not _TOOLS_REGISTRY: # Ensure tools are loaded if registry is empty
        load_all_tools()
        
    selected = []
    for name in names:
        tool = _TOOLS_REGISTRY.get(name)
        if tool:
            selected.append(tool)
        else:
            logger.warning(f"Tool not found: {name}")
    return selected


def get_tool(name: str) -> Optional[BaseTool]:
    if not _TOOLS_REGISTRY: # Ensure tools are loaded if registry is empty
        load_all_tools()
    return _TOOLS_REGISTRY.get(name)


def get_schema(name: str) -> Optional[Type[BaseModel]]:
    if not _TOOLS_REGISTRY: # Ensure tools are loaded if registry is empty
        load_all_tools()
    return _SCHEMA_REGISTRY.get(name)


def clear_registry() -> None:
    """
    Efface le registre d'outils.
    Utile pour les tests.
    """
    _TOOLS_REGISTRY.clear()
    _SCHEMA_REGISTRY.clear()
    logger.debug("Registre d'outils effacé")


def ensure_tools_imported() -> None:
    """
    S'assure que tous les modules d'outils sont importés pour enregistrer les outils.
    Cherche et importe dynamiquement tous les fichiers dans le dossier tools.
    """
    if _TOOLS_REGISTRY:
        return  # Les outils sont déjà chargés
    
    # Chemin du dossier des outils
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parcourir tous les fichiers Python dans le dossier tools
    for filename in os.listdir(tools_dir):
        if filename.endswith('.py') and filename != '__init__.py' and filename != 'registry.py':
            module_name = filename[:-3]  # Enlever l'extension .py
            module_path = f"app.tools.{module_name}"
            
            try:
                # Importer le module pour enregistrer les outils
                importlib.import_module(module_path)
                logger.debug(f"Module d'outils importé: {module_path}")
            except Exception as e:
                logger.error(f"Erreur lors de l'importation du module {module_path}: {str(e)}")
    
    # Log des outils enregistrés
    logger.info(f"Outils enregistrés: {list(_TOOLS_REGISTRY.keys())}")


# Importer automatiquement les outils au chargement du module
ensure_tools_imported() 