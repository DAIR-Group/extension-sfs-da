### How to Automatically Replace the Files  
If Using Anaconda
- First, initialize and activate the target environment (if you want to use your conda base environment, you should replace 'your-env-name' by 'base'):  
```bash
$ conda init
$ conda activate your-env-name
```

- Then, run the following command to replace the necessary files (file 'replace_files.py'):

``` bash
python replace_files.py --env anaconda --dir files_to_replace
```

If Using System Python (Non-Anaconda) - Run the following command:

```bash
$ python replace_files.py --env python --dir files_to_replace
```