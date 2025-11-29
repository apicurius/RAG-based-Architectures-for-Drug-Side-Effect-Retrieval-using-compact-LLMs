#!/usr/bin/env python3
import openpyxl
import itertools

wb = openpyxl.load_workbook('results_RAG_models_ALLdrugs_10randomADRs_llama38B.xlsx')
print('Sheet names:', wb.sheetnames)
ws = wb.active
print('\nFirst 10 rows:')
for row in itertools.islice(ws.rows, 10):
    print([cell.value for cell in row])
