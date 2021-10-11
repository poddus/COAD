
## Notes
* In order to maintain consistency in naming conventions, files and cases are referred to strictly by their UUID.
* The original plan was to use a single ```case_to_file``` dictionary to store all files for a given case, and output all data into a common directory. It has remained in place since it most naturally represents the data as a file belonging to a case, but its use creates more work. May be refactored at a later time.

## Set Flags


```python
first_run = False
verbose = True

verboseprint = print if verbose else lambda *a, **k: None
```

## Dependencies


```python
import os
import glob
import requests
import csv
import shutil
from collections import defaultdict
try:
    import simplejson as json
except ImportError:
    import json
import gzip

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from scipy import interp
from matplotlib.pyplot import get_cmap

from plotly.offline import download_plotlyjs
from plotly.offline import init_notebook_mode
from plotly.offline import iplot
import plotly.graph_objs as go
init_notebook_mode()

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

input_dir = os.path.join(os.getcwd(), 'in')
output_dir = os.path.join(os.getcwd(), 'out')

# initialize input directory
in_clin_manifest = os.path.join(
    input_dir,
    'clinical',
    'clinical_manifest.tsv')
in_clin_data = os.path.join(
    input_dir,
    'clinical',
    'data')
in_maf_file = os.path.join(input_dir,
  'maf',
  'TCGA.COAD.mutect.853e5584-b8c3-4836-9bda-6e7e84a64d97.DR-7.0.somatic.maf')
varsome_api_key_file = os.path.join(input_dir, 'varsome_api_key')

# initialize output directory
out_clin_data = os.path.join(output_dir, 'clinical', 'data')
out_clin_meta = os.path.join(output_dir, 'clinical', 'meta')
mut_df_file = os.path.join(output_dir, 'mut_df.gz')
trans_df_file = os.path.join(output_dir, 'trans_df.gz')

if first_run:
    for path in [out_clin_data, out_clin_meta]:
        os.mkdirs(path)
        
class tcgaFileQuery:
    def __init__(self, file_id):
        _cases_endpt = 'https://gdc-api.nci.nih.gov/files/ids'
        self._payload = {'query': file_id}
        self._response = requests.get(_cases_endpt, params=self._payload)
        self.data = self._response.json()

        self.case_id = self.data['data']['hits'][0]['cases'][0]['case_id']
        self.data_type = self.data['data']['hits'][0]['data_type']

def dump_json(json_data, path):
    with open(path, 'w') as jsonFile:
        json.dump(json_data, jsonFile, separators=(',', ': '), indent=4)
        
def fetch_metadata(manifest, output_location):
    file_to_case = {}
    UUID_to_filename = {}

    # metadata from server to file_to_case
    verboseprint('loading ' + manifest + '...')
    with open(manifest) as tsvFile:
        # get lenth of manifest file
        for line_count, value in enumerate(tsvFile):
            pass
        tsvFile.seek(0)  # return to top of file
        tsvData = csv.reader(tsvFile, delimiter='\t')

        next(tsvData, None)  # skip header row of manifest file
        for row in tsvData:
            file_id = row[0]
            UUID_to_filename[file_id] = row[1]
            
            verboseprint(
                str(line_count)
                + '\tfetching metadata for '
                + str(file_id)
                + '...',
                end=''
                )

            json_file = os.path.join(
                output_location,
                str(file_id) + '_meta.json')
            
            if first_run:
                queryResult = tcgaFileQuery(file_id)
                dump_json(queryResult.data, json_file)

               # store important metadata in databse
                file_to_case[file_id] = (queryResult.case_id,
                                         queryResult.data_type)

            else:
                with open(json_file, 'rt') as jsonFile:
                    reader = jsonFile.read()
                    jsonData = json.loads(reader)

                json_case_id = jsonData['data'][
                    'hits'][0]['cases'][0]['case_id']
                json_data_type = jsonData['data'][
                    'hits'][0]['data_type']

                # store important metadata in databse
                file_to_case[file_id] = json_case_id, json_data_type

            verboseprint('done')
            line_count -= 1

    verboseprint('all metadata retrieved successfully', '\n')

    verboseprint('flipping association...', end=' ')
    # flip association, since one case potentially
    # contains multiple files
    case_to_file = {}
    for file_id in file_to_case:
        case_id = file_to_case[file_id][0]
        file_type = file_to_case[file_id][1]
        case_to_file[case_id] = file_id, file_type
    verboseprint('done')

    return case_to_file, UUID_to_filename

def remove_annotated(input_data, case_to_file, UUID_to_filename):
    
    # by using a defaultdict, even if multiple files
    # of a case are annotated, no duplicate entries are
    # created (as with a list) and no key errors result
    is_annotated = defaultdict(lambda: False)

    for case_id, f in case_to_file.items():
        file_id = f[0]
        path_to_file = os.path.join(input_data,
                                    file_id,
                                    UUID_to_filename[file_id])
            
#         verboseprint('looking at file', file_id, '... ')
        if os.listdir(os.path.split(path_to_file)[
            0]).count('annotations.txt') > 0:
            verboseprint('annotation found for case', case_id)
            is_annotated[case_id] = True

    verboseprint(len(is_annotated), 'annotated cases found')

    verboseprint('ignoring annotated cases')
    for case in is_annotated.keys():
        del case_to_file[case]
    
    return case_to_file
```

## Retrieve from Disk
if the workflow has been executed before, the assembled DataFrame can be extracted from disk. Use the link below to jump to the Data Analysis section.


```python
case_to_mut_df = pd.read_pickle(mut_df_file, compression='gzip')
```


```python
case_to_trans_df = pd.read_pickle(trans_df_file, compression='gzip')
```

**[Jump to Exploration and Analysis](#Exploration-and-Analysis)**

# Genomics

### Fetch Genomic Metadata


```python
clin_c_to_f, clin_id_to_fn = fetch_metadata(in_clin_manifest, out_clin_meta)
```

### Remove annotated cases
cases are annotated mostly due to some confounders and would require more complicated analysis to compensate for, so we will ignore them for the time being


```python
clin_c_to_f = remove_annotated(in_clin_data, clin_c_to_f, clin_id_to_fn)
verboseprint('after removing annotated cases, our n =', len(clin_c_to_f))
```

### Copy Clinical Data to Output
this way we retain the input as-is and can work on the output without worrying too much.


```python
def clin_to_output():
    verboseprint('copying/extracting to output folder...')
    for case, file in clin_c_to_f.items():
        in_file = os.path.join(in_clin_data, file[0], clin_id_to_fn[file[0]])
        out_location = os.path.join(out_clin_data, case)
        out_file = os.path.join(out_clin_data, case, file[0] + '.xml')

        os.mkdir(out_location)
        verboseprint('copying file', file[0], 'to output...', end='')
        shutil.copyfile(in_file, out_file)
        verboseprint('done')

    verboseprint('extraction complete\n')
```


```python
if first_run: clin_to_output()
```

### Define "Left" & "Right"


```python
anatomic_right = ('Cecum',
                  'Ascending Colon',
                  'Hepatic Flexure')
                    # 'Transverse Colon'
anatomic_left = ('Descending Colon',
                 'Sigmoid Colon',
                 'Rectosigmoid Junction')
                    #  'Splenic Flexure'
```

### Extract Tumor Location
retrieve tumor location information from clinical data. some cases do not contain anatomic location information, so we cannot use them for our analysis


```python
def extract_tumor_location_from_clin():
    tumor_location = {}
    cases_without_location = []

    for case, file in clin_c_to_f.items():
        # only for clinical xml data
        file_loc = os.path.join(out_clin_data, case, file[0] + '.xml')
        with open(file_loc) as xmlFile:
            tree = ET.parse(xmlFile)
            root = tree.getroot()

            if root[1][34].text == None:
                cases_without_location.append(case)
            elif root[1][34].text in anatomic_right:
                tumor_location[case] = False
            elif root[1][34].text in anatomic_left:
                tumor_location[case] = True

    verboseprint(len(cases_without_location),
                 'cases do not contain anatomic location information')

    for case in cases_without_location:
        verboseprint('ignoring case', case)
        del clin_c_to_f[case]

    # return as pandas Series
    return pd.Series(tumor_location)
```


```python
tumor_location = extract_tumor_location_from_clin()
```

## The MAF File

### Format
* [Format specification](https://wiki.nci.nih.gov/display/TCGA/TCGA+MAF+Files)
* [TCGA search result for all COAD MAF files](https://portal.gdc.cancer.gov/repository?facetTab=files&filters=~%28op~%27and~content~%28~%28op~%27in~content~%28field~%27cases.project.project_id~value~%28~%27TCGA-COAD%29%29%29~%28op~%27in~content~%28field~%27files.data_format~value~%28~%27MAF%29%29%29%29%29)
* [Legacy Tumor-Normal Pair](https://portal.gdc.cancer.gov/legacy-archive/files/3437ecf9-355d-4d35-afb4-ffe1a705c206)

### The Problem
cells are defined by their genes. cancers carry variations in those genes which cause aberrant behavior. however, this causal relationship is not one-to-one, and additionally these variations do not need to be somatic mutations. they can be part of the individuals genome before the cancer develops. we require some function which is capable of separating irrelevant 'polymorphisms' from cancer-causing 'mutations'.

### The Solution
The GDCs approach is the MAF file. It compares the tumor genome to the individuals normal genome and a [reference sequence](https://www.ncbi.nlm.nih.gov/grc/human). The choice of reference sequence [can impact the results](https://www.biostars.org/p/113100/). There are multiple workflows which achieve this result

* varscan
* mutect
* muse
* somaticsniper

### HGVSc
* stands for "Human Genome Variation Society, coding". the full nomenclature definition can be found [here](http://varnomen.hgvs.org/recommendations/DNA/). there are following possible suffixes.
    * "g" genomic
    * "m" mitochondrial
    * "c" coding DNA
    * "n" non-coding DNA
    * "r" RNA reference sequence (transcript)
    * "p" protein reference sequence
* the HGVSc variant information is based off of reference sequences. these can be found in ```Gene```


```python
# columns of maf file (without loading entire file)
! echo $in_maf_file
! head -6 $in_maf_file | tail -1 | tr '\t' '\n'
```

### TODO: Reference_Allele vs HGVSc
in synonymous variant entries, the ```Reference_Allele``` vs ```Allele``` **differs** from the ```HGVSc``` notation!
* for example, for ```PERM1``` the alleles are ```G``` and ```A```, but the ```HGVSc``` notation specifies ```C>T```


```python
# mutect and somaticsniper share columns
maf_df = pd.read_csv(in_maf_file, sep='\t',
            usecols=[
                'Hugo_Symbol',
#                 'Reference_Allele',
                'HGVSc',
                'HGVSp',
#                 'Allele',
#                 'Gene',
                'Consequence',
                'case_id'
            ],
           header=5)
```


```python
maf_df.head()
```

### Intersection of Sets
some cases in the data portal do not have entries in the MAF file. analogously, some entries in the MAF file do not seem to be associated with any case in the current data portal version. we may only use the intersection of these two sets.

```tumor_location``` is derived from ```clin_c_to_f``` - ```clin_is_annotated``` - ```cases_without_location```.  
some of these cases are not present in the MAF file. therefore we cannot use them.


```python
bad_cases = []
for case in tumor_location.keys():
    if case not in maf_df['case_id'].unique():
        verboseprint('case', case, 'has no entry in MAF file')
        bad_cases.append(case)

tumor_location.drop(bad_cases, inplace=True)
del bad_cases
```


```python
# trimmed of cases that are not in MAF
tumor_location.shape[0]
```


```python
# maf not yet trimmed
len(maf_df['case_id'].unique())
```


```python
# for some cases in MAF we do not have adequate data.
# therefore we must remove them.
# drop entries where the case_id is NOT (~) in tumor_location.
entries_of_cases_without_location = maf_df.loc[~maf_df[
    'case_id'].isin(tumor_location.keys())]['case_id'].index
```


```python
maf_df.iloc[entries_of_cases_without_location].case_id.unique().shape[0]
```


```python
maf_df.drop(entries_of_cases_without_location, inplace=True)
```


```python
# number of cases matches trimmed location cases if == 0
len(maf_df['case_id'].unique()) - tumor_location.shape[0]
```


```python
maf_df['case_id'].unique().shape[0]
```

### Remove k- and n-RAS
* Mutations in these genes correlate strongly with resistance to anti-IgF therapy.
* Hugo Symbols are ```KRAS``` and ```NRAS```
* unfortunately, this reduces our n by about 50%


```python
kRAS_cases = maf_df.loc[maf_df['Hugo_Symbol'] == 'KRAS']['case_id']
nRAS_cases = maf_df.loc[maf_df['Hugo_Symbol'] == 'NRAS']['case_id']
```


```python
knRAS_cases_set = set(kRAS_cases.tolist() + nRAS_cases.tolist())
```


```python
len(knRAS_cases_set)
```


```python
entries_of_cases_with_knRAS_mutations = maf_df.loc[maf_df[
    'case_id'].isin(knRAS_cases_set)].index
```


```python
maf_df.drop(entries_of_cases_with_knRAS_mutations, inplace=True)
```


```python
maf_df['case_id'].unique().shape[0]
```

### Remove Synonymous Variants
using the ```Consequence``` column, value: ```synonymous_variant```


```python
synonymous_variants = maf_df.loc[
    maf_df['Consequence'] == 'synonymous_variant'].index
maf_df.drop(synonymous_variants, inplace=True)
```


```python
synonymous_variants.shape[0]
```

### Unique Variant Identifier
In order to completely define our mutations, we must combine multiple columns. A complete definition can be formatted as ```Gene```:```HGVSc```
* i.e. ```PERM1``` is ```ENSG00000187642:c.1827C>T```


```python
# create unique variant ID, store in column 'uvi'
# note, there are some NaN in the data set
maf_df.loc[:,'uvi'] = pd.Series(maf_df[
    'Hugo_Symbol'] + ':' + maf_df['HGVSc'], index=maf_df.index)
```


```python
case_to_mut_df = pd.DataFrame(False, index=maf_df[
    'case_id'].unique(), columns=maf_df.uvi.unique())
```


```python
# add column to end of DataFrame, left = True
case_to_mut_df.loc[:, 'tumor_loc_left'] = tumor_location
```


```python
for case, mutations in maf_df.groupby('case_id')['uvi']:
    for mut in mutations:
        case_to_mut_df.at[case, mut] = True
```


```python
def mut_freq_histogram():
    df_left = case_to_mut_df.loc[case_to_mut_df['tumor_loc_left'] == True]
    df_right = case_to_mut_df.loc[case_to_mut_df['tumor_loc_left'] == False]

    df_left.drop('tumor_loc_left', axis=1)
    df_right.drop('tumor_loc_left', axis=1)

    count_left = []
    for case in df_left.index:
        count_left.append(df_left.loc[case].value_counts()[1])

    count_right = []
    for case in df_right.index:
        count_right.append(df_right.loc[case].value_counts()[1])
    
    trace1 = go.Histogram(
        x=count_left,
        name='left-sided',
        opacity=0.8
        )

    trace2 = go.Histogram(
        x=count_right,
        name='right-sided',
        opacity=0.8
        )

    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay')
    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
```


```python
mut_freq_histogram()
```

#### Remove hypermutated cases
cases with more than 250 mutations are removed to approximately account for MSI status


```python
count = 0
for case in case_to_mut_df.index:
    if case_to_mut_df.loc[case].value_counts()[1] > 250:
        count += 1
        case_to_mut_df.drop(case, inplace=True)
print('removed', count, 'cases')
```


```python
case_to_mut_df.shape[0]
```

## Save to Disk
to save us the hassle of reevaluating every time, store finished, assembled DataFrame to disk.


```python
case_to_mut_df.to_pickle(mut_df_file, compression='gzip')
```

# Transcriptome
using transcriptome information from the TCGA Database, we can expand the amount of information per case.

possibilities include
* adding transcriptome information to the MAF Dataframe, thereby expanding the dimensionality of our data set
    * This could however be potentially problematic, because we would introduce continuous values into our previously purely dichotomous data. This could make the analysis more complicated.
* creating a separate database, running a logistic regression on that set, and comparing it to the genomic regression. In theory it should be possible to combine the results of both regressions to get a combined result.
* associating the variants with their corresponding transcript, and using some combination of this data to train our classifier.
    * It is unclear to me how the combined value should be calculated, and what it would represent. For example, if there are `20` variants in gene `x1` and the corresonding transcipt `x2` has a FPKM-UQ value of `1000` what would the function f(x1, x2), used to combine these two bits of information, look like?


```python
in_trans_manifest = os.path.join(
    input_dir,
    'transcriptome',
    'transcriptome_manifest.tsv')
in_trans_data = os.path.join(input_dir, 'transcriptome', 'data')

out_trans_data = os.path.join(output_dir, 'transcriptome', 'data')
out_trans_meta = os.path.join(output_dir, 'transcriptome', 'meta')
```


```python
trans_c_to_f, trans_id_to_fn = fetch_metadata(
    in_trans_manifest,
    out_trans_meta)
```


```python
trans_c_to_f = remove_annotated(
    in_trans_data,
    trans_c_to_f,
    trans_id_to_fn)
```


```python
len(trans_c_to_f)
```


```python
bad_cases = []
for case in trans_c_to_f.keys():
    if case not in case_to_mut_df.index:
        verboseprint('case', case, 'has no entry in MAF file')
        bad_cases.append(case)

verboseprint(len(bad_cases), 'cases will be removed from analysis')
for case in bad_cases:
    trans_c_to_f.pop(case)
del bad_cases
```


```python
len(trans_c_to_f)
```


```python
case_to_mut_df.shape
```


```python
def trans_extract_to_output():
    verboseprint('copying/extracting to output folder...')
    for case, file in trans_c_to_f.items():
        in_file = os.path.join(in_trans_data, file[0], trans_id_to_fn[file[0]])
        out_location = os.path.join(out_trans_data, case)
        out_file = os.path.join(out_trans_data, case, file[0] + '.txt')

        os.mkdir(out_location)
        verboseprint('copying file', file[0], 'to output...', end='')
        with gzip.open(in_file, 'rb') as f:
            file_content = f.read()
        with open(out_file, 'wb') as f:
            f.write(file_content)
        verboseprint('done')

    verboseprint('extraction complete\n')
```


```python
if first_run: trans_extract_to_output()
```


```python
for case in case_to_mut_df.index:
    try: trans_c_to_f[case]
    except KeyError:
        verboseprint(case)
        case_to_mut_df.drop(case, inplace=True)
```


```python
def make_transcriptome_DataFrame():
    transcriptome = defaultdict(dict)

    for case, file in trans_c_to_f.items():
        file_loc = os.path.join(out_trans_data, case, file[0] + '.txt')
        with open(file_loc, 'r') as tsvFile:
            tsvData = csv.reader(tsvFile, delimiter='\t')
            for row in tsvData:
                transcriptome[case][row[0]] = row[1]
    
    return pd.DataFrame.from_dict(transcriptome, orient='index', dtype='float')
```


```python
case_to_trans_df = make_transcriptome_DataFrame()
```


```python
case_to_trans_df.loc[:, 'tumor_loc_left'] = tumor_location
```

## Save to Disk
to save us the hassle of reevaluating every time, store finished, assembled DataFrame to disk.


```python
case_to_trans_df.to_pickle(trans_df_file, compression='gzip')
```

# Exploration and Analysis

### Naming Convention
`X` is the information we give the regression model, `y` is the correct result.

### t-Distributed Stochastic Neighbor Embedding
We can use t-SNE to visualize our dataset in 2 or 3 dimensions, to try to gain further insight
into our data set. It is useful for quickly reducing the dimensionality of a dataset
without any prior insight. In this example, 3 dimensions are used. this can be chosen
using the `n_components` variable. The model reduces high-dimensional datasets to something that
humans are capable of comprehending. May yield different results on repeated runs.

## Logistic Regression
A binary logistic regressor, using nested cross-validation.
* Outer CV
    * Sample Selection: Stratified k-fold
    * Scoring: **R**eceiver **O**perating **C**haracteristic **A**rea **U**nder the **C**urve
* Inner CV
    * Hyperparameter `C` is assessed for `n_inner_folds` (default: 10) logarithmically spaced values between 10<sup>-4</sup> and 10<sup>4</sup>
    * Sample Selection: Stratified k-fold
    * Scoring: Accuracy

## Random Forest
A binary classifier based on random subsampling and decision trees.


```python
def get_n_colors(n):
    """return list of colors from viridis colorspace for use with plotly"""
    
    cmap = get_cmap('viridis')
    colors_01 = cmap(np.linspace(0, 1, n))
    colors_255 = []

    for row in colors_01:
        colors_255.append(
            'rgba({}, {}, {}, {}'.format(
                row[0] * 255,
                row[1] * 255,
                row[2] * 255,
                row[3]
                )
            )

    return colors_255
```


```python
def tsne(df):

    tsne = TSNE(n_components=3, perplexity=20, n_iter=5000)

    X = df.drop('tumor_loc_left', axis=1)
    y = df['tumor_loc_left']

    X_3d = tsne.fit_transform(X)

    verboseprint('t-SNE')
    verboseprint('-----')
    verboseprint('')
    verboseprint('reduced dimensionality from {} to {} dimensions'.format(
        X.shape[1], X_3d.shape[1]
        )
    )

    trace_true = go.Scatter3d(
        x = X_3d[y==True, 0],
        y = X_3d[y==True, 1],
        z = X_3d[y==True, 2],
        name = 'left',
        mode = 'markers',
        marker = dict(
            color = 'rgba(37, 146, 34, 0.8)'
            ),
        text = X[y==True].index.tolist()
        )
    trace_false = go.Scatter3d(
        x = X_3d[y==False, 0],
        y = X_3d[y==False, 1],
        z = X_3d[y==False, 2],
        name = 'right',
        mode = 'markers',
        marker = dict(
            color = 'rgba(209, 28, 36, 0.8)'
            ),
        text = X[y==False].index.tolist()
        )
    layout = go.Layout(title='t-SNE Representation')

    fig = go.Figure(data=[trace_true, trace_false], layout=layout)

    iplot(fig)
```


```python
class logit_clf:
    def __init__(self, df, n_inner_folds=12, n_outer_folds=10, v=0):

        self.df = df
        self.n_outer_folds = n_outer_folds
        
        self.X = df.drop('tumor_loc_left', axis=1)
        self.y = df['tumor_loc_left']

        self.classifier = LogisticRegressionCV(
                                Cs=10,
                                fit_intercept=True,
                                cv=n_inner_folds,
                                dual=False,
                                penalty='l1',
                                solver='liblinear',
                                tol=0.0001,
                                max_iter=100,
                                class_weight=None,
                                n_jobs=-1,
                                verbose=v,
                                refit=True,
                                intercept_scaling=1.0,
                                random_state=3257
                                )
        self.cv = StratifiedKFold(
            n_splits=self.n_outer_folds,
            random_state=5235)

    def fit_and_print_roc(self):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        plotly_data = []

        colors = get_n_colors(self.n_outer_folds)
        
        i = 0
        for train, test in self.cv.split(self.X, self.y):
            probabilities_ = self.classifier.fit(
                                self.X.iloc[train],
                                self.y.iloc[train]
                                ).predict_proba(self.X.iloc[test])

            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve(
                self.y.iloc[test],
                probabilities_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plotly_data.append(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name='ROC fold {} (AUC = {})'.format(
                        i, round(roc_auc,2)),
                    line=dict(color=colors[i], width=1)
                    )
                )
            i += 1

        # add ROC reference line
        plotly_data.append(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(
                    color='navy',
                    width=2,
                    dash='dash'
                    ),
                showlegend=False
                )
            )

                # mean
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Standard Deviation
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plotly_data.append(
            go.Scatter(
                x=mean_fpr,
                y=tprs_upper,
                name='upper bound',
                line=dict(color='grey'),
                opacity=0.1,
                showlegend=False
                )
            )
        # plot mean above std deviation
        plotly_data.append(
            go.Scatter(
                x=mean_fpr,
                y=tprs_lower,
                name='± 1 std. dev.',
                fill='tonexty',
                line=dict(color='grey'),
                opacity=0.1
                )
            )

        plotly_data.append(
            go.Scatter(
                x=mean_fpr,
                y=mean_tpr,
                name='Mean ROC (AUC = {} ± {})'.format(
                    round(mean_auc, 2),
                    round(std_auc, 2)
                    ),
                line=dict(color='darkorange', width=3),
                )
            )
        
        layout = go.Layout(title='Receiver operating characteristic',
                   xaxis=dict(title='False Positive Rate'),
                   yaxis=dict(title='True Positive Rate')
                   )
        fig = go.Figure(data=plotly_data, layout=layout)

        iplot(fig)

    def get_n_pos_most_important(self, r_min, r_max, n=10, plot=True):
        """Return Variants whose Regression Coefficients are largest.

        assumes that the target class is `left`.
        """
        coefs = self.classifier.fit(self.X, self.y).coef_[0]
        indices = np.argsort(coefs)[::-1]

        n_indices = []
        for i in range(n):
            n_indices.append(indices[i])

        if plot:
            trace1 = go.Bar(x=self.df.columns[n_indices],
                           y=coefs[n_indices],
                           text = self.df.columns[n_indices],
                           marker=dict(color='green'),
                           opacity=0.5
                          )

            layout = go.Layout(
                title="Positive Regression Coefficients",
                yaxis=dict(range=[r_min, r_max])
            )
            fig = go.Figure(data=[trace1], layout=layout)

            iplot(fig)

    def get_n_neg_most_important(self, r_min, r_max, n=10, plot=True):
        """Return Variants whose Regression Coefficients are largest.

        assumes that the target class is `left`.
        """
        coefs = self.classifier.fit(self.X, self.y).coef_[0]
        indices = np.argsort(coefs)

        n_indices = []
        for i in range(n):
            n_indices.append(indices[i])

        if plot:
            trace1 = go.Bar(x=self.df.columns[n_indices],
                           y=coefs[n_indices],
                           text = self.df.columns[n_indices],
                           marker=dict(color='green'),
                           opacity=0.5
                          )

            layout = go.Layout(
                title="Negative Regression Coefficients",
                yaxis=dict(range=[r_min, r_max])
            )
            fig = go.Figure(data=[trace1], layout=layout)

            iplot(fig)

    def get_n_most_sided(self, n=10):
        """return two dicts, right and left, each with n case UUIDs
        and respective P(right-sidedness) - P(left-sidedness), where
        this value is minimal in the classifier.

        the classifier must be fit() before using this function.
        """

        # The order of the classes in probas corresponds to that
        # in the attribute classes_, in this case [False, True]

        probas = self.classifier.predict_proba(self.X)
        proba_diffs = np.array(probas[:,0] - probas[:,1])

        indices = np.argsort(proba_diffs)

        left_indices = []
        right_indices = []
        for i in range(n):
            left_indices.append(indices[i])
            right_indices.append(np.flip(indices, 0)[i])

        right = {}
        for i in right_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            right[case_id] = proba_diff

        left = {}
        for i in left_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            left[case_id] = proba_diff

        return right, left
```


```python
class rf_clf:
    def __init__(self, df, n_trees=1000, v=0):
        self.df = df
        self.X = df.drop('tumor_loc_left', axis=1)
        self.y = df['tumor_loc_left']
        self.classifier = RandomForestClassifier(
                                n_estimators=n_trees,
                                criterion='gini',
                                max_features='sqrt',
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                min_impurity_split=None,
                                bootstrap=True,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=3576,
                                verbose=v,
                                warm_start=False,
                                class_weight=None
                                )
    def fit(self):
        self.classifier.fit(self.X, self.y)

    def print_roc(self):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        y_decision = self.classifier.oob_decision_function_[:,1]
        fpr, tpr, thresholds = roc_curve(self.y, y_decision)
        roc_auc = auc(fpr, tpr)

        trace0 = go.Scatter(
            x=fpr,
            y=tpr,
            name='ROC AUC = {}'.format(round(roc_auc, 2)),
            line=dict(color='darkorange', width=2)
            )
        trace1 = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='navy', width=2, dash='dash'),
            showlegend=False
            )
        layout = go.Layout(
                    title='Receiver operating characteristic',
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate')
                    )

        data = [trace0, trace1]
        fig = go.Figure(data=data, layout=layout)

        iplot(fig)

    def get_n_most_important(self, n=10, plot=True):
        """Return Variants IDs deemed most important by the Classifier.

        Metric used by rf clf is (by default) `gini impurity`, or
        if specified, entropy gain.
        """
        def std_dev_of_features(n=8000):
            """use chunking to get std dev in memory-efficient way

            For large numbers of features and large numbers of trees,
            the memory required to store the input matrix for np.std becomes
            unfeasable. Here, the approach is to scan through all trees
            multiple times, selecting only a batch of features at a time
            and then calculating the std dev on that batch. This way we avoid
            loading the entire matrix of possibilities into memory at once.
            """

            std = np.zeros(self.X.shape[1])
            for i in range(0, self.X.shape[1], n):
                chunk_buffer = []
                for tree in self.classifier.estimators_:
                    chunk_buffer.append(tree.feature_importances_[i:i+n])
                std[i:i+n] = np.std(chunk_buffer, axis=0)
                del chunk_buffer

            return std

        std = std_dev_of_features()
        importances = self.classifier.feature_importances_
        indices = np.flip(np.argsort(importances), 0)

        n_indices = []
        for i in range(n):
            n_indices.append(indices[i])

        if plot:
            # plotly
            trace = go.Bar(x=self.df.columns[n_indices],
                           y=importances[n_indices],
                           text = self.df.columns[n_indices],
                           marker=dict(color='green'),
                           error_y=dict(
                               visible=True,
                               arrayminus=std[n_indices]),
                           opacity=0.5
                          )

            layout = go.Layout(title="Feature importance")
            fig = go.Figure(data=[trace], layout=layout)

            iplot(fig)
        
        return self.df.columns[n_indices].tolist()

    def get_n_most_sided(self, n=10):
        """return two dicts, right and left, each with n case UUIDs
        and respective P(right-sidedness) - P(left-sidedness), where
        this value is minimal in the classifier.

        the classifier must be fit() before using this function.
        """

        # The order of the classes in probas corresponds to that
        # in the attribute classes_, in this case [False, True]

        probas = self.classifier.predict_proba(self.X)
        proba_diffs = np.array(probas[:,0] - probas[:,1])

        indices = np.argsort(proba_diffs)

        left_indices = []
        right_indices = []
        for i in range(n):
            left_indices.append(indices[i])
            right_indices.append(np.flip(indices, 0)[i])

        right = {}
        for i in right_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            right[case_id] = proba_diff

        left = {}
        for i in left_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            left[case_id] = proba_diff

        return right, left
```


```python
tsne(case_to_trans_df)
```

    t-SNE
    -----
    
    reduced dimensionality from 60483 to 3 dimensions


```python
mut_logit_clf = logit_clf(case_to_mut_df)
mut_logit_clf.fit_and_print_roc()
mut_logit_important = mut_logit_clf.get_n_most_important()
mut_logit_sided = mut_logit_clf.get_n_most_sided()
```


```python
forest_size = case_to_mut_df.shape[1] // 4
mut_rf_clf = rf_clf(case_to_mut_df, n_trees=forest_size, v=1)
mut_rf_clf.fit()
mut_rf_clf.print_roc()
mut_rf_important = mut_rf_clf.get_n_most_important()
mut_rf_sided = mut_rf_clf.get_n_most_sided()
```


```python
trans_logit_clf = logit_clf(case_to_trans_df, v=1)
trans_logit_clf.fit_and_print_roc()
trans_logit_important = trans_logit_clf.get_n_most_important()
trans_logit_sided = trans_logit_clf.get_n_most_sided()
```


```python
forest_size = case_to_trans_df.shape[1] // 4
trans_rf_clf = rf_clf(case_to_trans_df, v=1)
trans_rf_clf.fit()
trans_rf_clf.print_roc()
trans_rf_important = trans_rf_clf.get_n_most_important()
trans_rf_sided = trans_rf_clf.get_n_most_sided()
```

**TODO** we can get the `n` cases which best represent left- or right-sidedness by finding `P(left) - P(right)`. Then, we take the case UUIDs and download the slides from the legacy TCGA. Prof. Kirchner can then typify them.


```python
import pickle
```

## Save Classifiers to Disk


```python
with open('./out/classifiers/mut_logit_clf.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(mut_logit_clf, f, pickle.HIGHEST_PROTOCOL)
```


```python
with open('./out/classifiers/mut_rf_clf.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(mut_rf_clf, f, pickle.HIGHEST_PROTOCOL)
```


```python
with open('./out/classifiers/trans_logit_clf.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(trans_logit_clf, f, pickle.HIGHEST_PROTOCOL)
```


```python
with open('./out/classifiers/trans_rf_clf.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(trans_rf_clf, f, pickle.HIGHEST_PROTOCOL)
```

## Retrieve Classifiers from Disk


```python
# Mutation Logistic Regression
with open('./out/classifiers/mut_logit_clf.pickle', 'rb') as f:
    unpickler = pickle.Unpickler(f)
    mut_logit_clf = unpickler.load()
```


```python
# Mutation RandomForest
with open('./out/classifiers/mut_rf_clf.pickle', 'rb') as f:
    unpickler = pickle.Unpickler(f)
    mut_rf_clf = unpickler.load()
```


```python
# Transcriptome Logistic Regression
with open('./out/classifiers/trans_logit_clf.pickle', 'rb') as f:
    unpickler = pickle.Unpickler(f)
    trans_logit_clf = unpickler.load()
```


```python
# Transcriptome RandomForest
with open('./out/classifiers/trans_rf_clf.pickle', 'rb') as f:
    unpickler = pickle.Unpickler(f)
    trans_rf_clf = unpickler.load()
```


```python
mut_rf_clf.classifier.predict_proba(
    case_to_mut_df.drop(
        'tumor_loc_left', axis=1))[:,0]
```


```python
mut_logit_clf.get_n_pos_most_important(r_min=0, r_max=8)
```


```python
mut_logit_clf.get_n_neg_most_important(r_min=-8, r_max=0)
```


```python
trans_logit_clf.get_n_pos_most_important(r_min=0, r_max=0.000004)
```


```python
trans_logit_clf.get_n_neg_most_important(r_min=-0.000004, r_max=0)
```


```python
np.array(
    mut_rf_clf.classifier.predict_proba(
        case_to_mut_df.drop(
            'tumor_loc_left', axis=1))[
        :,0] - mut_rf_clf.classifier.predict_proba(
        case_to_mut_df.drop('tumor_loc_left', axis=1))[:,1]
)
```

## Variant Information
* using the [VarSome](https://api.varsome.com/) search engine API, we can retrieve a large amount of information about our variants.
* [Reference API Client](https://github.com/saphetor/variant-api-client-python) written in Python, hosted on GitHub

#### Available GET Parameters
params are passed as dict with key value pairs, for example  
```params={'expand-pubmed-articles': 1, 'add-source-databases': 'gerp,wustl-civic'}```

* ```add-all-data``` = 1 or 0 
* ```add-region-databases``` = 1 or 0 
* ```expand-pubmed-articles``` = 1 or 0 
* ```add-main-data-points``` = 1 or 0 
* ```add-source-databases``` = all or none or
    * gerp
    * wustl-civic
    * ncbi-dbsnp
    * ensembl-transcripts
    * broad-exac
    * dbnsfp-dbscsnv
    * gwas
    * ncbi-clinvar
    * gnomad-genomes
    * dbnsfp
    * sanger-cosmic
    * dann-snvs
    * sanger-cosmic-public
    * gnomad-exomes
    * ncbi-clinvar2
    * refseq-transcripts
    * uniprot-variants
    * gnomad-genomes-coverage
    * gnomad-exomes-coverage
    * thousand-genomes
    * isb-kaviar3
    * iarc-tp53-germline
    * sanger-cosmic-licensed
    * iarc-tp53-somatic
    * icgc-somatic
* ```allele-frequency-threshold``` = float


```python
from variantapi.client import VariantAPIClient
varsome_api_key = with open(varsome_api_key_file) as f: f.read()
varsome_api = VariantAPIClient(varsome_api_key)
```


```python
result = varsome_api.batch_lookup(
    list(significant_var_dict.keys()),
    params={'add-source-databases': 'gerp'},
    ref_genome='hg19')
```


```python
result
```

### various test snippets


```python
missence_variants = maf_df.loc[
    maf_df['Consequence'] == 'missense_variant'].index
var_list = []
for entry in maf_df.loc[missence_variants][
    ['Hugo_Symbol', 'HGVSc']].values:
    # concatenate as HUGO:HGVSc
    var_def = entry.tolist()[0] + ':' + entry.tolist()[1]
    var_list.append(var_def)

# create batches of at most n variants
n = 10
q_chunks = [var_list[i:i+n] for i in range(
    0, len(var_list), n)]
```

**TODO: above, n is set to 10 for testing, for final release, set to 10000 (ten thousand)**


```python
result = varsome_api.batch_lookup(q_chunks[0],
                    params={'add-source-databases': 'gerp'},
                    ref_genome='hg19')
```


```python
interesting = ['alt', 'chromosome']
```


```python
[result[0][x] for x in interesting]
```
