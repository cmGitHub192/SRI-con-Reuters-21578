from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

# Ruta para procesar la query
@app.route('/process', methods=['POST'])
def process():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    script_path = os.path.join(os.getcwd(), 'backend', 'retrieval.py')
    result = subprocess.run(['python',script_path , query], capture_output=True, text=True)

    if result.returncode != 0:
        return jsonify({'error': 'Processing failed', 'details': result.stderr}), 500

    return jsonify({'result': result.stdout})

if __name__ == '__main__':
    app.run(debug=True)
