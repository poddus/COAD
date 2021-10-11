import json
import requests


class CasesQuery:
    def __init__(self):
        _cases_endpt = 'https://api.gdc.cancer.gov/cases/'
        self._payload = {
            'filters': {
                'op': 'in',
                'content': {
                    'field': 'cases.project.project_id',
                    'value': ['TCGA-COAD']
                }
            }
        }
        self._response = requests.post(_cases_endpt, json=self._payload)
        self.data = self._response.text


class TestQuery:
    def __init__(self, payload):
        endpoint = 'https://api.gdc.cancer.gov/files/'
        self._payload = payload
        self._response = requests.post(endpoint, json=self._payload)
        self.data = self._response.text


clinical_payload = {
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
    'fields': 'cases.case_id'
}

transcriptome_payload = {
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
    'fields': 'cases.case_id'
}

test = TestQuery(transcriptome_payload)
print(test._response.url)
with open('../json_dump.json', 'w') as f:
    json.dump(test._response.json(), f)
pass
