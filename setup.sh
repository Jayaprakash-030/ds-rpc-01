#!/bin/bash

echo "🚀 [1/7] Setting up Git identity..."
git config --global user.name "Jayaprakash"
git config --global user.email "jayprakeshsai@gmail.com"
git config --global credential.helper store

echo "📦 [2/7] Installing Python requirements..."
pip install -r requirements.txt

echo "🧠 [3/7] Installing development tools for IntelliSense..."
pip install ipykernel python-dotenv

echo "📁 [4/7] Creating VS Code config folders..."
mkdir -p .vscode
mkdir -p ~/.vscode-server/data/Machine

echo "⚙️ [5/7] Writing VS Code workspace settings..."
cat <<EOF > .vscode/settings.json
{
  "python.languageServer": "Pylance",
  "python.analysis.indexing": true,
  "python.analysis.autoSearchPaths": true,
  "python.analysis.useLibraryCodeForTypes": true,
  "python.defaultInterpreterPath": "/usr/bin/python3"
}
EOF

echo "🧩 [6/7] Adding recommended extensions..."
cat <<EOF > .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "esbenp.prettier-vscode"
  ]
}
EOF

echo "🔐 [7/7] Disabling VS Code signature verification (for remote install fix)..."
cat <<EOF > ~/.vscode-server/data/Machine/settings.json
{
  "extensions.verifySignature": false
}
EOF

echo "✅ Setup complete!"
echo "💡 Next steps:"
echo "   1. In VS Code, press Cmd+Shift+P → 'Reload Window' to apply all settings."
echo "   2. Go to Cmd+Shift+P → 'Python: Select Interpreter' and choose: /usr/bin/python3 (Remote)."
echo "   3. You can now install extensions in SSH without signature errors."

# chmod +x setup.sh
# ./setup.sh
# huggingface-cli login
