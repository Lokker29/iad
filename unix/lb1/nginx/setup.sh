cd nginx-1.19.9 || exit 1

CONF_PATH=/etc/nginx/nginx.conf
DEFAULT_VARS_PATH=/etc/default/nginx

./configure --conf-path=$CONF_PATH --error-log-path=/var/logs/error.log --http-log-path=/var/logs/access.log
make
make install

cp ../nginx.conf $CONF_PATH
touch $DEFAULT_VARS_PATH
echo "NGINX_CONF_FILE=$CONF_PATH" > $DEFAULT_VARS_PATH

echo "SUCCESS SETUP"
