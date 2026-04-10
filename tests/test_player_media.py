from deuce.player_media import extract_wta_headshot_url


def test_extract_wta_headshot_url_prefers_player_image():
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://photoresources.wtatennis.com/photo-resources/2025/05/06/home-share.png" />
      </head>
      <body>
        <img alt="Sabalenka-Torso_320760" src="https://photoresources.wtatennis.com/photo-resources/2024/04/22/sabalenka-torso.png?width=790&height=740" />
      </body>
    </html>
    """
    assert extract_wta_headshot_url(html) == "https://photoresources.wtatennis.com/photo-resources/2024/04/22/sabalenka-torso.png?width=790&height=740"


def test_extract_wta_headshot_url_ignores_generic_share_image():
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://photoresources.wtatennis.com/photo-resources/2025/05/06/Home-Share_1200x630.png?width=1200&height=630" />
      </head>
      <body></body>
    </html>
    """
    assert extract_wta_headshot_url(html) is None
