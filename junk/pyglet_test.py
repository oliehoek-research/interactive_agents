# Test using pyglet within WSL - doing policy visualization with pyglet
import pyglet


def main():
    config = pyglet.gl.Config(sample_buffers=1, samples=4)
    window = pyglet.window.Window(500, 500, config=config)

    shapes = pyglet.graphics.Batch()
    triangle = shapes.add(3, pyglet.gl.GL_TRIANGLES, None,
            ('v2i', (200, 200, 300, 200, 250, 300)),
            ('c3B', (255, 0, 0, 0, 255, 0, 0, 0, 255))
        )

    label = pyglet.text.Label(f"A Triangle",
                              font_name="Arial",
                              font_size=16,
                              x=250, y=50,
                              anchor_x="center", anchor_y="center",
                              color=(255,255,255,255), bold=True)

    @window.event
    def on_draw():
        window.clear()
        shapes.draw()

        label.draw()

    pyglet.app.run()


if __name__ == "__main__":
    main()
