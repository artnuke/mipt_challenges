# How to create custome EvalAI challenges.

!!!IMPORTANT!!! Read original README written by EvalAI developers!

## Edit "evaluate" function. 
Importent nuances:
- "evaluate" is the main method
- Add all requirements in the ini.py file.
- Submission file path stores in "user_submission_file" varible.
- Data file stores in annotation folder And avalible by path stored in the "user_submission_file".
- The function must return result in following format "return result['result'] = [{ <Here should be your output data> }]". Output data must match leaderbord split and phase data split nottation.

## Upadte challenge_config file with your challenge configuration.
Importent nuances:
- Keep in mind that all codenames, names, pathes e.t.c must match in all sections of the file.

## Upload challenge and restart worker.
Notes: 
- After aploading you have to aprove challenge within the admin panel.
- You should restart worker service.
'''
sudo docker-compose restart worker
'''
