# Test using pyglet within WSL - doing policy visualization with pyglet
import pyglet

if __name__ == "__main__":
    # NOTE: There is a full game tutorial for pyglet, maybe go through that

    config = pyglet.gl.Config(sample_buffers=1, samples=4)
    window = pyglet.window.Window(500, 500, config=config)

    shapes = pyglet.graphics.Batch()
    triangle = shapes.add(3, pyglet.gl.GL_TRIANGLES, None,
            ('v2i', (200, 200, 300, 200, 250, 300)),
            ('c3B', (255, 0, 0, 0, 255, 0, 0, 0, 255))
        )

    @window.event
    def on_draw():
        window.clear()
        shapes.draw()

    pyglet.app.run()  # NOTE: How do we manually increment the game loop?
