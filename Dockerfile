###############
# BUILD IMAGE #
###############

FROM python:3.9.5-slim

# set working directory
WORKDIR $Udacity_DS_ND_Capstone

COPY . . /

RUN ls -la $Udacity_DS_ND_Capstone/


# streamlit-specific commands

RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

EXPOSE 8501

#ADD . /Udacity_Capstone_DS_NDp

#COPY requirement.txt /tmp/

RUN pip install --upgrade pip==21.1.2
# RUN pip install --requirement /tmp/requirement.txt
RUN pip install --requirement requirement.txt


# Set the locale
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# Run streamlit
CMD ["streamlit", "run", "--server.enableCORS", "false", "web_app.py"]