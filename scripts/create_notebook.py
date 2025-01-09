import nbformat as nbf

nb = nbf.v4.new_notebook()

# Add markdown cells
nb["cells"] = [
    nbf.v4.new_markdown_cell("# Devin Proto Mabataki - デモ\n\nアニメキャラクターの目パチ・口パクアニメーション生成エンジンのデモノートブックです。"),
    nbf.v4.new_markdown_cell("## セットアップ\n\n必要なライブラリをインストールします。"),
    nbf.v4.new_code_cell("!git clone https://github.com/yuiseki/devin-proto-mabataki.git\n%cd devin-proto-mabataki\n!pip install -r requirements.txt"),
    nbf.v4.new_markdown_cell("## アプリケーションの起動\n\nGradioインターフェースを起動します。"),
    nbf.v4.new_code_cell("from src.app import MabatakiApp\n\napp = MabatakiApp()\ninterface = app.create_interface()\ninterface.launch()")
]

# Create notebooks directory if it doesn't exist
import os
os.makedirs("notebooks", exist_ok=True)

# Write the notebook
with open("notebooks/demo.ipynb", "w") as f:
    nbf.write(nb, f)
