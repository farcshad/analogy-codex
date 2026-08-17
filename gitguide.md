git push when somebody else has pushed something:
you get ERROR!!!
you have to first pull. and sometimes due the conflicts you cant. 


when you want to close the laptop and get the thing to be ran while you sleep.
create tmux!!!!
what is tmux (?)

creation: tmux new -s {name of the session}
list of sessions: tmux ls
how to go back to the normal terminal(?): control+b (release) then d


when you want to continue on a tmux from before:
attach to a session: tmux attach -t {name of the session}
close a tmux:
detach from a session: tmux detach


have a snapshot view of the session: tmux capture-pane -pt {name}


how to kill a tmux session: tmux kill-session -t {name}


how to run the code in tmux? while you are in tmux terminal:

first activate venv: copy venv's path (i.e. users/analogy/venv) 
then type : source /users/analogy/venv/bin/activate

then you can run the code.

how? in case of notebooks. put all the codes together in one cell. 
example: 

codeline1
codeline2
codeline3


you need this command:

python -c "
codeline1
codeline2
codeline3
"

then you copy this command and enter it in tmux terminal.




nvidia-smi
how to kill all the gpu sessions
kill -9 python