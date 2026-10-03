from groq import Groq
clien=Groq
models=clien.models.list()
for m in models.data:
    print(m.id)