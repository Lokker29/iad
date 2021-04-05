apt-get update && apt-get install wget build-essential libpcre3-dev zlib1g-dev nano -y

wget http://nginx.org/download/nginx-1.19.9.tar.gz
tar -xsf nginx-1.19.9.tar.gz
rm nginx-1.19.9.tar.gz

wget https://raw.githubusercontent.com/JasonGiedymin/nginx-init-ubuntu/master/nginx -P /etc/init.d
chmod +x /etc/init.d/nginx

echo "SUCCESS INSTALL"
