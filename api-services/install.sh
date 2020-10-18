if ! [ -x "$(command -v pipenv)" ]; then
        echo 'Warning: pipenv is not installed.' >&2
        pip3 install pipenv
    else
        echo 'Error: pipenv is installed.' >&2
fi

if ! [ -x "$(command -v flask)" ]; then
        echo 'Warning: flask is not installed.' >&2
        # pip3 install flask
  else
        echo 'Error: flask is installed.' >&2
fi


if ! [ -x "$(command -v joblib)" ]; then
        echo 'Warning: joblib is not installed.' >&2
        # pip3 install joblib
    else
        echo 'Error: joblib is installed.' >&2
fi

if ! [ -x "$(command -v torch)" ]; then
        echo 'Warning: torch is not installed.' >&2
        # pip3 install torch
    else
        echo 'Error: torch is installed.' >&2
fi

if ! [ -x "$(command -v numpy)" ]; then
        echo 'Warning: numpy is not installed.' >&2
        # pip3 install numpy
    else
        echo 'Error: numpy is installed.' >&2
fi

if ! [ -x "$(command -v scipy)" ]; then
        echo 'Warning: scipy is not installed.' >&2
        # pip3 install scipy
    else
        echo 'Error: scipy is installed.' >&2
fi

if ! [ -x "$(command -v sklearn)" ]; then
        echo 'Warning: sklearn is not installed.' >&2
        pip3 install sklearn
    else
        echo 'Error: sklearn is installed.' >&2
fi

# postgresql
#  postgres /var/lib/postgresql/13/main /var/log/postgresql/postgresql-13-main.log
if ! [ -x "$(command -v psql )" ]; then
        echo 'Warning: postgresql is not installed.' >&2
        sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
        wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
        sudo apt-get update
        sudo apt-get -y install postgresql
  else
        echo 'Error: postgresql is installed.' >&2
        echo 'Start portgresql...' >&2
        # start postgresql
        # https://manpages.debian.org/buster/postgresql-common/pg_ctlcluster.1.en.html
        sudo pg_ctlcluster 13 main start
fi
