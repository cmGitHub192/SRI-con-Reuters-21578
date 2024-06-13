import requests
import json
import os
import re

def extract_numbers_from_response(api_response):
    numbers = re.findall(r'\d+', api_response)
    return [int(number) for number in numbers]

def test_api(query):
    url = 'http://127.0.0.1:5000/process'
    headers = {'Content-Type': 'application/json'}

    payload = {'query': query}
    numbers_set = set()

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        data_str = json.dumps(data)  # Convertir a cadena de texto
        numbers = extract_numbers_from_response(data_str)
        numbers_set = set(numbers)
    
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud para query '{query}':", e)
    
    return numbers_set



def read_queries_from_file(file_path):
    queries = []
    with open(file_path, 'r') as file:
        for line in file:
            # Dividir cada línea en dos partes: nombre de la consulta y números
            parts = line.strip().split(': ')
            if len(parts) == 2:
                query_name = parts[0]
                numbers = set(map(int, parts[1].split(', ')))
                queries.append((query_name, numbers))
    return queries

def calculo_recall(predichos, gt):
    TP = len(predichos.intersection(gt))
    FP = len(predichos.difference(gt))
    total = TP + FP
    
    if total == 0:
        return 0.0  # Evita división por cero
    
    recall = TP / total
    return recall * 100

def calculo_precision(predichos, gt):
    TP = len(predichos.intersection(gt))
    FN = len(gt.difference(predichos))
    total = TP + FN
    
    if total == 0:
        return 0.0  # Evita división por cero
    
    precision = TP / total
    return precision * 100


if __name__ == '__main__':
    file_path = os.path.join(os.getcwd(),'..', 'Proyecto_BIM_I', 'reuters', 'indice_invertido.txt')
    queries = read_queries_from_file(file_path)
    for query_name, numbers_set in queries:
        results = test_api(query_name)
        print("====================================================================")
        print("Resultados para: ", query_name)
        print("Cantidad de predicciones: ", len(results))
        print("Recall: ", calculo_recall(results, numbers_set))
        print("Precision: ", calculo_precision(results, numbers_set))

