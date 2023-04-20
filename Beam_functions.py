'''All the functions needed for notebook'''

import numpy as np
from scipy.optimize import fmin, minimize, LinearConstraint
import plotly.graph_objects as go
import panel as pn
pn.extension()
pn.extension('plotly')

# Units definition for more clarity 
cm = 1.0
mm = 1e-1*cm
um = 1e-4*cm
nm = 1e-7*cm

m = 100*cm

def waist_bennink(wavelength, crystal_length, n, xi=2.84):
    k = 2*np.pi*n/wavelength
    return np.sqrt(crystal_length / (k * xi))

def beam_constants(n_p_crystal, n_p_air, crystal_length, init_waist, wl):
    '''Function for putting all the initial constants into one variable'''
    const = (n_p_crystal,n_p_air,crystal_length,init_waist,wl)
    return const

def propagation(d):
    # Using `dtype=object` to shut up some stupid warning ;)
    return np.array([[1, d], [0, 1]], dtype=object)

def planar_refraction(n1, n2):
    """Note: n1 - initial, n2 - final."""
    return np.array([[1, 0], [0, n1/n2]])

def thin_lens(f):
    return np.array([[1, 0], [-1/f, 1]])

def transform_waist(waist_in, transformation, wavelength, n=1):
    A, B = transformation[0]
    k = 2 * np.pi * n / wavelength
    waist_out = np.sqrt(A**2 * waist_in**2 + 4 * B**2 / (k**2 * waist_in**2))
    return waist_out

def pump_beam(z, constants, focusing_lens=None, lens_to_crystal=None):
    
    N_P_CRYSTAL = constants[0]
    N_P_AIR = constants[1]
    CRYSTAL_LENGTH = constants[2]
    INIT_WAIST = constants[3]
    WL = constants[4] 
    
    if focusing_lens:
        focusing_lens_pos, focusing_lens_f = focusing_lens
    init_waist_pos, init_waist_width = INIT_WAIST
    
    # Free propagation: from "zero" to focusing lens (if any)
    if not focusing_lens or z <= focusing_lens_pos:
        s = propagation(z - init_waist_pos)
        w = transform_waist(init_waist_width, s, WL)
        return w

    # Through the focusing lens and propagate to crystal (if any)
    s = propagation(focusing_lens_pos - init_waist_pos)
    z -= focusing_lens_pos
    if not lens_to_crystal or z <= lens_to_crystal - CRYSTAL_LENGTH / 2:
        s = propagation(z) @ thin_lens(focusing_lens_f) @ s
        w = transform_waist(init_waist_width, s, WL)
        return w
    
    # Refract on the first face of the crystal and propagate inside
    s = propagation(lens_to_crystal - CRYSTAL_LENGTH / 2) @ thin_lens(focusing_lens_f) @ s
    z -= lens_to_crystal - CRYSTAL_LENGTH / 2
    if z <= CRYSTAL_LENGTH:
        s = propagation(z) @ planar_refraction(N_P_AIR, N_P_CRYSTAL) @ s
        w = transform_waist(init_waist_width, s, WL)
        return w
    
    # Finally, refraction at the second face of the crystal and free propagation
    s = propagation(CRYSTAL_LENGTH) @ planar_refraction(N_P_AIR, N_P_CRYSTAL) @ s
    z -= CRYSTAL_LENGTH
    s = propagation(z) @ planar_refraction(N_P_CRYSTAL, N_P_AIR) @ s
    w = transform_waist(init_waist_width, s, WL)
    return w

def waist_in_crystal(x, focusing_lens_pos, focusing_lens_f, const):
    """Return waist pos in crystal midpoint. A helper function. Minimize this wrt `lens_to_crystal` """
    z, lens_to_crystal = x
    waist = pump_beam(z, focusing_lens=[focusing_lens_pos, focusing_lens_f], lens_to_crystal=lens_to_crystal, constants = const)
    return waist

def find_pump_waist_in_crystal(focusing_lens, const, tol=1e-12):
    
    focusing_lens_pos, focusing_lens_f = focusing_lens
    expected_waist_pos = focusing_lens_pos + focusing_lens_f
    waist_bounds = (focusing_lens_pos, focusing_lens_pos + 2 * focusing_lens_f)
    expected_lens_to_crystal = focusing_lens_f
    
    sol = minimize(waist_in_crystal, [expected_waist_pos, expected_lens_to_crystal],
                   args=(focusing_lens_pos, focusing_lens_f, const),
                   constraints=({"type": "eq", "fun": lambda x: x[0] - x[1] - focusing_lens_pos}),
                   bounds=[waist_bounds, (None, None)], tol=tol)
    sol_waist_pos, sol_lens_to_crystal = sol["x"]
    sol_waist_width = sol["fun"]
    return sol_waist_pos, sol_waist_width, sol_lens_to_crystal

def fig_graph_test(wavelength, crystal_length, crystal_width, refractive_index, lens_position, focal_length, pump_waist, crystal_position, choice, width, height):
    
    wl = wavelength*nm
    crystal_length = crystal_length*mm
    crystal_width = crystal_width*mm
    
    crystal_position = crystal_position*mm + focal_length
    f_lens = (lens_position*cm, focal_length*cm)
    
    PUMP_WAIST_WIDTH = pump_waist*cm # Initial pump beam width
    PUMP_WAIST_POS = 0*cm  #Waist position is at the collimator at position x = 0.
    
    const = beam_constants(n_p_crystal=refractive_index, n_p_air=1, crystal_length=crystal_length, init_waist=(PUMP_WAIST_POS, PUMP_WAIST_WIDTH), wl=wl)
    # Length to crystal (ltc) should be set as focal distance from lens
    ltc = crystal_position #f_lens[1]
    end = f_lens[0] + ltc + 20*cm
    zs = np.linspace(0, end, 1000)
    
    #ws_nocrystal = np.array([pump_beam(z, const, focusing_lens=f_lens) for z in zs])
    ws = np.array([pump_beam(z, const, focusing_lens=f_lens, lens_to_crystal=ltc) for z in zs])
    
    intersections = np.where(np.around(ws,3) == crystal_width/2) # find indexes where the beam width is equal to crystal width
    focus_point = np.where(zs <= f_lens[0] + f_lens[1]) # 
    
    insec1 = intersections[0][intersections[0] < focus_point[0][-1]]
    insec2 = intersections[0][intersections[0] > focus_point[0][-1]]

    crystal_start = np.where(np.around(zs,1) >= f_lens[0] + ltc - crystal_length/2)
    crystal_end = np.where(np.around(zs,1) <= f_lens[0] + ltc + crystal_length/2)
    crystal_int = np.intersect1d(crystal_start,crystal_end)
    
    k = 2*np.pi*refractive_index/wl
    ksi = round(crystal_length/(k*min(ws*cm)**2),2)
    
    position_x = 0.5
    position_y = 0.85
    
    if choice == []:
        if insec1[0] <= crystal_int[0] and insec2[-1] >= crystal_int[-1]:
            
            line = go.Scatter(x=zs, y=ws,name= str(wavelength) + ' nm laser',showlegend=True,marker_color='rgba(255, 0, 0, 0.8)')
            
            layout = go.Layout(autosize=False, width=width, height=height,
                  title='Laser beam focusing in crystal',
                  xaxis_title='Distance [cm]',
                  yaxis_title='Beam waist [cm]',
                  yaxis_zeroline=False, xaxis_zeroline=True,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  annotations=[dict(xref="x domain",yref="y domain", x=1,y=1, text='<b>Ksi parameter</b>: '+ str(ksi), font=dict(family="Arial", size=18, color = "white"), 
                                    bgcolor="#ff7f0e", opacity=1, showarrow=False, align="right")],   
                  shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                          dict(type="line", x0=f_lens[0], y0=0, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                          dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=0, x1=(f_lens[0] + ltc) + crystal_length/2, 
                               y1=crystal_width/2, line_width=0, fillcolor="blue", opacity=0.3)]
                          )
            fig = dict(data=[line], layout=layout)
        
        else:
           
            line_out = go.Scatter(x=zs, y=ws,name= str(wavelength) + ' nm laser', showlegend=True, marker_color='rgba(255, 0, 0, 0.2)')
                        
            layout_out = go.Layout(autosize=False, width=width, height=height,
                  title='Laser beam focusing in crystal',
                  xaxis_title='Distance [cm]',
                  yaxis_title='Beam waist [cm]',
                  yaxis_zeroline=False, xaxis_zeroline=True,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  annotations=[dict(xref="x domain", yref="y domain", x=position_x, y=position_y, text='<b>Beam clipping!</b>', font=dict(family="Arial", size=24, color = "red"), 
                                    opacity=0.8, showarrow=False, align="center")],  
                  shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                          dict(type="line", x0=f_lens[0], y0=0, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                          dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=0, x1=(f_lens[0] + ltc) + crystal_length/2, 
                               y1=crystal_width/2, line_width=0, fillcolor="blue", opacity=0.3)]
                          )
            fig = dict(data=[line_out], layout=layout_out)
            
    
    if choice == ['Symetric']:
        if insec1[0] <= crystal_int[0] and insec2[-1] >= crystal_int[-1]:
            
            line = go.Scatter(x=zs, y=ws,name= str(wavelength) + ' nm laser',showlegend=True,marker_color='rgba(255, 0, 0, 0.8)')
            
            line2 = go.Scatter(x=zs, y=-ws, showlegend=False, marker_color='rgba(255, 0, 0, 0.8)')
     
            layout_sym = go.Layout(autosize=False, width=width, height=height,
                                        title='Laser beam focusing in crystal',
                                        xaxis_title='Distance [cm]',
                                        yaxis_title='Beam waist [cm]',
                                        yaxis_zeroline=False, xaxis_zeroline=True,
                                        annotations=[dict(xref="x domain",yref="y domain", x=1,y=1, text='<b>Ksi parameter</b>: '+ str(ksi), font=dict(family="Arial", size=18, color = "white"), 
                                                        bgcolor="#ff7f0e", opacity=0.8, showarrow=False, align="right")], 
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                                                dict(type="line", x0=f_lens[0], y0=-PUMP_WAIST_WIDTH-PUMP_WAIST_WIDTH/10, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                                                dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=-crystal_width/2, x1=(f_lens[0] + ltc) + crystal_length/2, y1=crystal_width/2, 
                                                     line_width=0, fillcolor="blue", opacity=0.3)]
                                    )
            fig = dict(data=[line,line2], layout=layout_sym)
            
        else:
            line_out_sym = go.Scatter(x=zs, y=ws,name= str(wavelength) + ' nm laser', showlegend=True, marker_color='rgba(255, 0, 0, 0.2)')
            
            line2_out_sym = go.Scatter(x=zs, y=-ws, showlegend=False, marker_color='rgba(255, 0, 0, 0.2)')
            
            layout_out_sym = go.Layout(autosize=False, width=width, height=height,
                                        title='Laser beam focusing in crystal',
                                        xaxis_title='Distance [cm]',
                                        yaxis_title='Beam waist [cm]',
                                        yaxis_zeroline=False, xaxis_zeroline=True,
                                        annotations=[dict(xref="x domain", yref="y domain", x=position_x, y=position_y, text='<b>Beam clipping!</b>', font=dict(family="Arial", size=24, color = "red"), 
                                                    opacity=0.8, showarrow=False, align="center")],
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                                                dict(type="line", x0=f_lens[0], y0=-PUMP_WAIST_WIDTH-PUMP_WAIST_WIDTH/10, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                                                dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=-crystal_width/2, x1=(f_lens[0] + ltc) + crystal_length/2, y1=crystal_width/2, 
                                                     line_width=0, fillcolor="blue", opacity=0.3)]
                                    )
            fig = dict(data=[line_out_sym,line2_out_sym], layout=layout_out_sym)
    
    if choice == ['Zoom']:
        if insec1[0] <= crystal_int[0] and insec2[-1] >= crystal_int[-1]:
            
            line = go.Scatter(x=zs, y=ws,name= str(wavelength) + ' nm laser', showlegend=True, marker_color='rgba(255, 0, 0, 0.8)')
            
            layout_zoom = go.Layout(autosize=False, width=width, height=height,
                  title='Laser beam focusing in crystal',
                  xaxis_title='Distance [cm]',
                  yaxis_title='Beam waist [cm]',
                  yaxis_zeroline=False, xaxis_zeroline=True,
                  xaxis_range=[focal_length + lens_position - crystal_length, focal_length + lens_position + crystal_length],
                  yaxis_range=[-crystal_width*1.2, crystal_width*1.2],
                  annotations=[dict(xref="x domain",yref="y domain", x=1,y=1, text='<b>Ksi parameter</b>: '+ str(ksi), font=dict(family="Arial", size=18, color = "white"), 
                                    bgcolor="#ff7f0e", opacity=0.8, showarrow=False, align="right")], 
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                          dict(type="line", x0=f_lens[0], y0=-PUMP_WAIST_WIDTH-PUMP_WAIST_WIDTH/10, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                          dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=-crystal_width/2, x1=(f_lens[0] + ltc) + crystal_length/2, y1=crystal_width/2, line_width=0, fillcolor="blue", opacity=0.3)]
                          )
                
            fig = dict(data=[line], layout=layout_zoom)
        
        else:
            line_out_zoom = go.Scatter(x=zs, y=ws,
                      name= str(wavelength) + ' nm laser',
                      showlegend=True,
                      marker_color='rgba(255, 0, 0, 0.2)')
            
            layout_out_zoom = go.Layout(autosize=False, width=width, height=height,
                  title='Laser beam focusing in crystal',
                  xaxis_title='Distance [cm]',
                  yaxis_title='Beam waist [cm]',
                  yaxis_zeroline=False, xaxis_zeroline=True,
                  xaxis_range=[focal_length + lens_position - crystal_length, focal_length + lens_position + crystal_length],
                  yaxis_range=[-crystal_width*1.2, crystal_width*1.2],
                  annotations=[dict(xref="x domain", yref="y domain", x=position_x, y=position_y, text='<b>Beam clipping!</b>', font=dict(family="Arial", size=24, color = "red"), 
                                    opacity=0.8, showarrow=False, align="center")],
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                          dict(type="line", x0=f_lens[0], y0=-PUMP_WAIST_WIDTH-PUMP_WAIST_WIDTH/10, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                          dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=-crystal_width/2, x1=(f_lens[0] + ltc) + crystal_length/2, y1=crystal_width/2, line_width=0, fillcolor="blue", opacity=0.3)]
                          )
            fig = dict(data=[line_out_zoom], layout=layout_out_zoom)
    
    if choice == ['Symetric', 'Zoom']:
        if insec1[0] <= crystal_int[0] and insec2[-1] >= crystal_int[-1]:
            
            line_symzoom = go.Scatter(x=zs, y=ws,
                      name= str(wavelength) + ' nm laser',
                      showlegend=True,
                      marker_color='rgba(255, 0, 0, 0.8)')
            
            line2_symzoom = go.Scatter(x=zs, y=-ws, showlegend=False, marker_color='rgba(255, 0, 0, 0.8)')
        
            layout_symzoom = go.Layout(autosize=False, width=width, height=height,
                  title='Laser beam focusing in crystal',
                  xaxis_title='Distance [cm]',
                  yaxis_title='Beam waist [cm]',
                  xaxis_range=[focal_length + lens_position - crystal_length, focal_length + lens_position + crystal_length],
                  yaxis_range=[-crystal_width*1.2, crystal_width*1.2],
                  yaxis_zeroline=False, xaxis_zeroline=True,
                  annotations=[dict(xref="x domain",yref="y domain", x=1,y=1, text='<b>Ksi parameter</b>: '+ str(ksi), font=dict(family="Arial", size=18, color = "white"), 
                                    bgcolor="#ff7f0e", opacity=0.8, showarrow=False, align="right")], 
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                          dict(type="line", x0=f_lens[0], y0=-PUMP_WAIST_WIDTH-PUMP_WAIST_WIDTH/10, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                          dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=-crystal_width/2, x1=(f_lens[0] + ltc) + crystal_length/2, y1=crystal_width/2, line_width=0, fillcolor="blue", opacity=0.3)]
                          )
            fig = dict(data=[line_symzoom,line2_symzoom], layout=layout_symzoom)
        
        else:
            line_out_symzoom = go.Scatter(x=zs, y=ws,
                      name= str(wavelength) + ' nm laser',
                      showlegend=True,
                      marker_color='rgba(255, 0, 0, 0.2)')
            
            line2_out_symzoom = go.Scatter(x=zs, y=-ws, showlegend=False, marker_color='rgba(255, 0, 0, 0.2)')
            
            layout_out_symzoom = go.Layout(autosize=False, width=width, height=height,
                  title='Laser beam focusing in crystal',
                  xaxis_title='Distance [cm]',
                  yaxis_title='Beam waist [cm]',
                  xaxis_range=[focal_length + lens_position - crystal_length, focal_length + lens_position + crystal_length],
                  yaxis_range=[-crystal_width*1.2, crystal_width*1.2],
                  yaxis_zeroline=False, xaxis_zeroline=True,
                  annotations=[dict(xref="x domain", yref="y domain", x=position_x, y=position_y, text='<b>Beam clipping!</b>', font=dict(family="Arial", size=24, color = "red"), 
                                    opacity=0.8, showarrow=False, align="center")],
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  shapes=[dict(type="line", x0=0, y0=0, x1=end, y1=0, line_width=1),
                          dict(type="line", x0=f_lens[0], y0=-PUMP_WAIST_WIDTH-PUMP_WAIST_WIDTH/10, x1=f_lens[0], y1=PUMP_WAIST_WIDTH+PUMP_WAIST_WIDTH/10, line_width=3),
                          dict(type="rect", x0=(f_lens[0] + ltc) - crystal_length/2, y0=-crystal_width/2, x1=(f_lens[0] + ltc) + crystal_length/2, y1=crystal_width/2, line_width=0, fillcolor="blue", opacity=0.3)]
                          )
            fig = dict(data=[line_out_symzoom,line2_out_symzoom], layout=layout_out_symzoom)
          
    ptly_pane = pn.pane.Plotly(fig)
    return ptly_pane