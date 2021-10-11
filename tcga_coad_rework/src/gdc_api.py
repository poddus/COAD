import json
from typing import Tuple, List, Dict
import requests
import logging

CLINICAL_PAYLOAD = {
    'filters': {
        'op': 'and',
        'content': [
            {
                'op': 'in',
                'content': {
                    'field': 'cases.project.project_id',
                    'value': ['TCGA-COAD']
                }
            },
            {
                'op': 'in',
                'content': {
                    'field': 'files.data_type',
                    'value': [
                        'Clinical Supplement',

                    ]
                }
            }
        ]
    },
    'fields': 'cases.case_id',
    'size': '1000'
}
TRANSCRIPTOME_PAYLOAD = {
    'filters': {
        'op': 'and',
        'content': [
            {
                'op': 'in',
                'content': {
                    'field': 'cases.project.project_id',
                    'value': ['TCGA-COAD']
                }
            },
            {
                'op': 'in',
                'content': {
                    'field': 'files.analysis.workflow_type',
                    'value': ['HTSeq - FPKM-UQ']
                }
            }
        ]
    },
    'fields': 'cases.case_id',
    'size': '1000'
}


class GDCQuery:
    def __init__(self, payload):
        endpoint = 'https://api.gdc.cancer.gov/files/'
        self.payload = payload
        self.response = requests.post(endpoint, json=self.payload)
        self.data = self.response.text


def get_case_association(payload: Dict) -> Dict:
    query = GDCQuery(payload)
    logging.debug('API Query: {}'.format(query.response.request.url))
    decoded = json.loads(query.data)
    association = {}
    for hit in decoded['data']['hits']:
        association[hit['cases'][0]['case_id']] = hit['id']
    return association
