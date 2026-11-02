"""Centralized helper to suppress VTK popup windows, dialogs, and console warnings."""

import os
import vtk


def _patch_vedo_metrics() -> None:
    """Prevent vedo's area() and volume() from re-enabling SetGlobalWarningDisplay(1)."""
    try:
        import vedo

        if getattr(vedo, "_cbsr_metrics_patched", False):
            return

        orig_area = vedo.Mesh.area
        orig_volume = vedo.Mesh.volume

        def safe_area(self, *args, **kwargs):
            prev = vtk.vtkObject.GetGlobalWarningDisplay()
            res = orig_area(self, *args, **kwargs)
            if not prev:
                vtk.vtkObject.GlobalWarningDisplayOff()
            return res

        def safe_volume(self, *args, **kwargs):
            prev = vtk.vtkObject.GetGlobalWarningDisplay()
            res = orig_volume(self, *args, **kwargs)
            if not prev:
                vtk.vtkObject.GlobalWarningDisplayOff()
            return res

        vedo.Mesh.area = safe_area
        vedo.Mesh.volume = safe_volume
        vedo._cbsr_metrics_patched = True
    except Exception:
        pass


def silence_vtk() -> None:
    """Completely suppress VTK warnings, logger output, and Win32 popup windows."""
    try:
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    except Exception:
        pass

    try:
        vtk.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass

    try:
        # Suppress any existing output window prompt/display
        ow = vtk.vtkOutputWindow.GetInstance()
        if ow:
            ow.PromptUserOff()
            if hasattr(ow, "SetDisplayModeToNever"):
                ow.SetDisplayModeToNever()
    except Exception:
        pass

    try:
        # Redirect all VTK output window text away from vtkWin32OutputWindow to devnull
        file_out = vtk.vtkFileOutputWindow()
        file_out.SetFileName(os.devnull)
        file_out.PromptUserOff()
        if hasattr(file_out, "SetDisplayModeToNever"):
            file_out.SetDisplayModeToNever()
        vtk.vtkOutputWindow.SetInstance(file_out)
    except Exception:
        pass

    _patch_vedo_metrics()


# Run silencing immediately on import
silence_vtk()
