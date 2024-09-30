FROM debian:stretch

# install debian packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq \
    && apt-get install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    openssh-client \
    wget \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install python=3.7.9 && \
    pip install numpy --no-cache-dir && \
    pip install ipywidgets --no-cache-dir && \
    pip install requests --no-cache-dir && \
    pip install ipykernel --no-cache-dir && \
    pip install ipywidgets --no-cache-dir && \
    pip install plotly --no-cache-dir && \
    /opt/conda/bin/conda install nodejs

# Create a directory to mount homework at run 
RUN mkdir hw1

# Set the working directory to /hw1
WORKDIR /hw1

EXPOSE 8889

# Install the package and run ipython in no-browser mode
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --port=8889"]