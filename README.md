# ADMETlab
A platform for systematic ADME evaluation of drug molecules, thereby accelerating the drug discovery process.<br>
The platform is avaliable at: http://admet.scbdd.com 
(Note: the pgp-substrate model have been fixed, also the correct model has been updated in webserver)
## Features
1. Well optimized SAR models with better performance than state-of-the-art models.
2. User-friendly interface and easy to use.
3. Support batch computation.
4. Systematically evaluate molecular druggability.
5. Provide constructive suggestions for molecular optimization.

## About this repository.
This repository provides models of ADMETlab in binary files for expert users. The ordinary users are suggested to use the ADMETlab server to accomplish all related prediction or analysis tasks because of the convenience and no programming requirements.

The expert users could download these models to carry out further research. To use the model, download all the zipped files and unzip, then calculate the descriptors according to the "[Documentation](http://admet.scbdd.com/home/interpretation/)" section, then load the model to predict. The example folder contains an example. In the command line, input:
<br>
> python run.py
<br>
and you could get the results.

## About the algorithms and explanation.
The modeling process including descriptors, methods, feature selection and model performance has been detailedly described in the "[Documentation](http://admet.scbdd.com/home/interpretation/)" section of the website.

## Contact.
If you have questions or suggestions, please contact: jie@csu.edu.cn and oriental-cds@163.com

Please see the file LICENSE for details about the "New BSD" license which covers this software and its associated data and documents.
