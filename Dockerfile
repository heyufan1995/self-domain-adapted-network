FROM nvcr.io/nvidian/pytorch:20.03-py3
RUN pip install nibabel
RUN pip install monai
RUN pip install scikit-image
RUN pip install tifffile