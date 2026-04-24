set -ex

export COMMIT_SHA=$(git rev-parse --short HEAD)

docker compose -f swarm-config.yml build --push

docker -H ssh://root@moshi-chat.kyutai.org stack deploy -c swarm-config.yml --with-registry-auth moshi
