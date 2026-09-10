from vedo import Plotter, Cube, Plane, Points

viewer = Plotter(title="Multimedia Retrieval CBSR", size=(1000, 750), bg="black")

def on_scroll(interactor, event):
    factor = 1.15 if event == "MouseWheelForwardEvent" else 1 / 1.15

    camera = viewer.camera

    if camera.GetParallelProjection():
        camera.SetParallelScale(camera.GetParallelScale() / factor)
    else:
        camera.Dolly(factor)

    viewer.renderer.ResetCameraClippingRange()
    viewer.render()


viewer.interactor.AddObserver("MouseWheelForwardEvent", on_scroll)
viewer.interactor.AddObserver("MouseWheelBackwardEvent", on_scroll)

ground = Plane(
    pos=(0, 0, 0),
    normal=(0, 0, 1),
    s=(6,6)
).color("gray").alpha(0.3)

cube_model = Cube().color("lightblue")
cube_model.linewidth(1).linecolor("black")
cube_vertices = Points(cube_model.coordinates, r=5, c="red")
# cube_model.wireframe()

viewer.show(
    cube_model, ground, cube_vertices, 
    axes=8, 
    mode=8,
    viewup="z",
    interactive=True
)
viewer.close()