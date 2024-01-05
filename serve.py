from flask import Flask, send_from_directory
import os
app = Flask(__name__)

MEDIA_PATH = './'
play_viedeo="""
<script type=text/javascript>
   const movie_name = {{ movie_name|tojson }};
</script>


  <video id="video" defaultMuted autoplay playsinline controls>
    <source src="{{ url_for('media_video', filename=movie_name) }}" type="video/{{movie_ext}}">
    Your browser does not support the video tag.
  </video>
"""

@app.route('/')
def video_list():
    video_files = [os.path.join(root, f) for root, dirs, files in os.walk(MEDIA_PATH) for f in files if f.endswith('.webm')]
    # Inline HTML template
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video List</title>
    </head>
    <body>
        <h1>Video List</h1>
        <ul>
            {% for video_file in video_files %}
                <li><a href="{{ url_for('send_media', path=video_file) }}">{{ video_file }}</a></li>
            {% endfor %}
        </ul>
    </body>
    </html>
    """
    return app.jinja_env.from_string(template).render({'video_files': video_files})


@app.route('/media/<path:path>')
def send_media(path):
    """
    :param path: a path like "posts/<int:post_id>/<filename>"
    """
    print(path)
    return send_from_directory(directory="./", path=path, as_attachment=False,conditional=False)

# @app.route("/playvideourl/<filename>")
# def playvideourl(filename): 
#     template="""
# <script type=text/javascript>
#    const movie_name = {{ movie_name|tojson }};
# </script>


#   <video id="video" defaultMuted autoplay playsinline controls>
#     <source src="{{ url_for('send_media', path=movie_name) }}" type="video/mp4">
#     Your browser does not support the video tag.
#   </video>
# """

#     return app.jinja_env.from_string(template).render({'movie_name': "test.mp4"})

if __name__ == '__main__':
    app.run(debug=True,port=1221)
