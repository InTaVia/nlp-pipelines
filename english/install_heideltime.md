In order to use Heideltime directly inside a Python Script we are using a third-party python library which needs a couple of extra steps to be properly setup.

1. Clone Python-Heideltime Repository
```
git clone git@github.com:PhilipEHausner/python_heideltime.git
```

2. If you are in linux run the provided script to install heideltime. Then go to step 5.
First replace line 15 of the `install_heideltime_standalone.sh` with this line: `wget --no-verbose https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.5.tar.gz`
Then Run:
```
cd python_heideltime
chmod +x install_heideltime_standalone.sh
./install_heideltime_standalone.sh
```

3. If you are on a Mac you need to change manually one line of the `install_heideltime_standalone.sh` script before running: 
```
wget --no-verbose https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-MacOSX-3.2.3.tar.gz
```

4. Additionally you must change some paths manually after running the script from step 3. Inside the python_heideltime directory, go to the file `heideltime-standalone/heideltime-standalone/config.props` and in line 24 put the absolute path to TreeTagger, for example:

```
/Users/daza/nlp-pipelines/python_heideltime/heideltime-standalone/treetagger
```

5. Now that heideltime is setup, make sure you are in the folder `python_heideltime` and install the python package:

```
python3 -m pip install .
```


6. Come back to our project root `nlp-pipelines` and test if Heideltime is with:

```
python3 english/utils/nlp_heideltime.py
```

the following output should appear:

```
<?xml version="1.0"?>
<!DOCTYPE TimeML SYSTEM "TimeML.dtd">
<TimeML>
<TIMEX3 tid="t1" type="DATE" value="2023-06-20">Yesterday</TIMEX3>, I bought a cat! It was born <TIMEX3 tid="t3" type="DATE" value="2023" mod="START">earlier this year</TIMEX3>.
</TimeML>
```