Here's your text formatted in Markdown for better readability:
To generate a `requirements.txt` file that includes only the top-level dependencies (i.e., the packages you have explicitly installed) without including the dependencies of those dependencies, you can use the `pipreqs` tool. This tool scans your project directory for imports and generates a `requirements.txt` file based on the packages you directly use in your code.

## Steps to Achieve This:

### 1. Install pipreqs:
``` bash
pip install pipreqs
```

### 2. Generate `requirements.txt`:
Navigate to the root directory of your project and run:
``` bash
pipreqs . --force
```
The `--force` flag ensures that any existing `requirements.txt` file is overwritten.

### Example
Assume your project directory structure is as follows:
```
my_project/
    ├── main.py
    ├── module1.py
    └── module2.py
```
If `main.py` and `module1.py` import packages like `numpy` and `pandas`, `pipreqs` will detect these imports and generate a `requirements.txt` file with only `numpy` and `pandas` listed.

## Steps in Detail

### 1. Install pipreqs:
``` bash
pip install pipreqs
```
### 2. Navigate to your project directory:
``` bash
cd path/to/your/project
```
### 3. Generate `requirements.txt`:
``` bash
pipreqs . --force
```
## Alternative Method: Using pipdeptree
Another method is to use `pipdeptree` to list the dependencies and then filter out the top-level packages. Here’s how you can do it:

### 1. Install pipdeptree:
``` bash
pip install pipdeptree
```
### 2. Generate a list of top-level packages:

``` bash
pipdeptree --freeze --warn silence | grep -E '^\w+' > requirements.txt
```
This command lists the dependencies in a tree format and then filters out the top-level packages using `grep`.

### Example Using pipdeptree
#### 1. Install pipdeptree:
``` bash
pip install pipdeptree
```
#### 2. Generate `requirements.txt`:
``` bash
pipdeptree --freeze --warn silence | grep -E '^\w+' > requirements.txt
```
This will create a `requirements.txt` file with only the top-level dependencies, avoiding the clutter of including all transitive dependencies.

By using either `pipreqs` or `pipdeptree`, you can generate a clean `requirements.txt` file that includes only the packages you have explicitly installed.
This Markdown format uses headers, code blocks, and lists to enhance clarity and organization.