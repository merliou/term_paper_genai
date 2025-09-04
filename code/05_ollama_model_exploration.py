import ollama

# Step 1: List all installed models
models = ollama.list()
print("Installed models:")
for model in models['models']:
    print("-", model['model'])

    