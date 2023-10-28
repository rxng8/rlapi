
import sys

def check_vscode_interactive() -> bool:
  return hasattr(sys, 'ps1')



