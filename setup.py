from setuptools import setup, find_packages

setup(name='robosteer12idb',
      version="0.1",
      description='Automated ML prediction of user arm positioning via coarse scans.', 
      packages = find_packages(
        where='robosteer12idb'
      ) ,
      install_requires = ['mlflow',
                          'matplotlib',
                          'torch',
                          'tqdm'
                 ],
      )