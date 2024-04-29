# Install the RL virtual-env, activate it and run this from shell:
rlOwnDir=$(dirname $(readlink -f "$BASH_SOURCE"))
rlBaseDir=$(dirname "$rlOwnDir")
export PYTHONPATH="${rlBaseDir}/src:${rlBaseDir}/test:${PYTHONPATH}" 
RUNPYTHON=python3

${RUNPYTHON} "${rlBaseDir}/test/test_rl/test_tennis/testAnimation.py" "$@"
