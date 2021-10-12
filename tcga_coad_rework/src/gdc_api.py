import json
from typing import Tuple, List, Dict
import requests
import logging

from src import config
from src.cache import read_api_response_from_cache, write_api_response_to_cache

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
    if config.USE_CACHED_API_RESPONSE:
        if payload == CLINICAL_PAYLOAD:
            return read_api_response_from_cache('clin_response')
        elif payload == TRANSCRIPTOME_PAYLOAD:
            return read_api_response_from_cache('trans_response')

    query = GDCQuery(payload)
    logging.debug('API Query: {}'.format(query.response.request.url))
    decoded = json.loads(query.data)
    association = {}
    for hit in decoded['data']['hits']:
        association[hit['cases'][0]['case_id']] = hit['id']

    if config.UPDATE_CACHE:
        if payload == CLINICAL_PAYLOAD:
            write_api_response_to_cache(association, 'clin_response')
        elif payload == TRANSCRIPTOME_PAYLOAD:
            write_api_response_to_cache(association, 'trans_response')
    return association
