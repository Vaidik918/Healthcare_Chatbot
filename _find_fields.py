import sys
files = ['check.py', 'chatbot.py']
for fname in files:
    print('=== ' + fname + ' ===')
    with open(fname, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if 'phone' in line.lower() or 'website' in line.lower():
                print(str(i) + ': ' + line.rstrip())
    print()
