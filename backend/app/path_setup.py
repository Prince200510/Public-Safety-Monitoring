from __future__ import annotations
import os
import sys

def ensure_workspace_on_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  
    backend_dir = os.path.dirname(here) 
    root_dir = os.path.dirname(backend_dir)  

    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    return root_dir
