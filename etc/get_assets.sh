cd assets/models

if [ ! -f sponza.zip ]; then
    wget -O sponza.zip http://themaister.net/sponza-gltf-pbr/sponza-gltf-pbr.zip
fi

rm -rf sponza
unzip sponza.zip
mv sponza-gltf-pbr sponza
