Comment rajouter un model :

Premierement copier le lien du model sur Huggingface que vous trouverez ici ->  ![alt text](/workspaces/GPT-Local-Q-A/images/Capture d’écran 2023-12-18 155421.png)

Ensuite mettez votre lien dans le fichier constant en le nommant comment vous le souhaiter -> ![alt text](/workspaces/GPT-Local-Q-A/images/Capture d’écran 2023-12-19 135751.png)

Dans le fichier Txt_qa.py rajouter une fonction comme celle si comprenant votre model -> ![alt text](/workspaces/GPT-Local-Q-A/images/Capture d’écran 2023-12-19 140257.png)

Dans le fichier streamlit_app_blog.py rajouter un elif avec votre model dans la fonction load_llm -> ![alt text](/workspaces/GPT-Local-Q-A/images/Capture d’écran 2023-12-19 140539.png)

Rajouter votre model dans le bouton -> ![alt text](/workspaces/GPT-Local-Q-A/images/Capture d’écran 2023-12-19 140649.png)

Enjoy




