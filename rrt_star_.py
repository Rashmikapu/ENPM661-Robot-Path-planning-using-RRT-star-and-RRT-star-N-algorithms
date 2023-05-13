def rrt_star(initial , final , canvas):
    start_time = time.time()
    T = []
   
    Td = {}
    
    node_dict = {}
    
    L = 20
    T.append(initial)
    
    Td[tuple(initial)] = initial
    
    while (1) :
        Zrand = Sample(i, canvas)
        
        print(Zrand)
        Znearest = Nearest(T , Zrand)                 #finding nearest node to the randomly generated node
        

        Znew  = Steer(Znearest, Zrand, L)          #creating a new node which is at distance of step size (or higher) from nearest node
        # T.append(Znew)    
                   
        # Td[tuple(Znew)] = Znearest 
        #Znew= Zrand
        if Obstacle(Znew, canvas) :
            continue
        
       
        radius = 5
            #Znear = Near(T , Znew, radius)        
        cost = Euclidean_dist(Znew, final) + Euclidean_dist(Znew, Znearest)
       
        node_dict[tuple(Znew)] = [Znearest , cost]    #appending unrewired parent node
        
        neighbors = generate_neighbors(Znew, T, radius)
        
        for j in neighbors :
            Znearest = rewire(j, Znew, node_dict, canvas, initial, final)

        
            
            
        # if edge_points(Znew, Znearest, canvas):
        #     print(f"{Znew} is rejected")
        #     continue   
            
        # if(collision_free(Znearest, Znew, canvas) is False):
        #     continue    

        T.append(Znew)    
                   
        Td[tuple(Znew)] = Znearest                         #new node : parent
        
        
        # for n in neighbors:
        #     X_parent_temp = rewire(Znew, n, node_dict, canvas, initial, final)

        #     if X_parent_temp == Znew:
        #         print("Before Pop", n)
        #         Td.pop(tuple(n))
        #         Td[tuple(n)] = Znew.copy()
        
        if goal_thresh(Znew, final) :
            print('GOAL FOUND')
            end_time = time.time()
            elapsed_time = end_time-start_time
            print(f"Time elapsed for RRT-star : {elapsed_time}")
            Backtrack1(Td, T, initial, Znew, canvas)
            break

# def rrt_star_n(initial , final , canvas):
#     start_time = time.time()
#     T = []
   
#     Td = {}
    
#     node_dict = {}
    
#     L = 30
#     T.append(initial)
    
#     Td[tuple(initial)] = initial
    
#     while (1) :
#         Zrand = nSample(i,initial,final,canvas)
        
#         print(Zrand)
#         Znearest = Nearest(T , Zrand)                 #finding nearest node to the randomly generated node
        

#         Znew  = Steer(Znearest, Zrand, L)          #creating a new node which is at distance of step size (or higher) from nearest node
        
#         #Znew= Zrand
#         if Obstacle(Znew, canvas) :
#             continue
        
       
#         radius = 5
#             #Znear = Near(T , Znew, radius)        
#         cost = Euclidean_dist(initial,Znew) + Euclidean_dist(Znew, final) + Euclidean_dist(Znew, Znearest)
       
#         node_dict[tuple(Znew)] = [Znearest , cost]    #appending unrewired parent node
        
#         neighbors = generate_neighbors(Znew, T, radius)
        
#         for j in neighbors :
#             Znearest = rewire(j, Znew, node_dict, canvas, initial, final)

        
            
            
        # if edge_points(Znew, Znearest, canvas):
        #     print(f"{Znew} is rejected")
        #     continue   
            
        # if(collision_free(Znearest, Znew, canvas) is False):
        #     continue    

        T.append(Znew)    
                   
        Td[tuple(Znew)] = Znearest                         #new node : parent
        
        
#    
        
        if goal_thresh(Znew, final) :
            print('GOAL FOUND')
            end_time = time.time()
            elapsed_time = end_time-start_time
            print(f"Time elapsed for RRT-star-n : {elapsed_time}")
            Backtrack1(Td, T, initial, Znew, canvas)
            break      


def nSample(i,initial,final,canvas2):
   
    combined_list = []
    gaussian= np.random.multivariate_normal(final,[[100000,-150000],[-150000,100000]],size = 1000)
    for i in gaussian:
        xg,yg = int(i[0]), int(i[1])
    
        if xg >= 0 and xg <= 1000 and yg >= 0 and yg <= 1000:
            combined_list.append((xg,yg))


    i = random.randint(0,len(combined_list)-1)
    #j = random.randint(0 , len(combined_list)-1)
    points = list(combined_list)
    return [points[i][1], points[i][0]]
    # points=[]
    # x1,y1 = initial[0], initial[1]
    # x2,y2 = final[0], final[1]
    # min_x = min(x1,x2)
    # min_y = min(y1,y2)
    # max_x = max(x1,x2)
    # max_y = max(y1,y2)
    # if x2-x1 !=0:
    #     m = (y2-y1)/(x2-x1)
    #     c = y1-m*x1
    #     for x in range(min_x,max_x):
    #         for y in range(min_y,max_y):
    #             if abs(y-m*x-c) < 1.5 :
    #                 points.append([x,y])

    # else :
    #     x = x1
    #     for j in range(min_y, max_y):
    #         points.append([x1,j])
    # print(f"POINTS:{points}")
    # min_px = points[0][0]
    # for i,j in points:
    #     if i < min_px :
    #         min_px = i
    # max_px = points[0][0]
    # for i,j in points:
    #     if i > max_px :
    #         max_px = i

    # min_py = points[0][0]
    # for i,j in points:
    #     if j< min_py :
    #         min_py = j

    # max_py = points[0][0]
    # for i,j in points:
    #     if j >min_px :
    #         max_py = j
            
    # i = random.randint(min_px-125,max_px+125)
    # j = random.randint(min_py-125 , max_py+125)
    # return [i,j]
#-----------------------------------------------------------------------
    # n = random.randint(0, len(points)-1)
    # i = points[n][0]
    # j = points[n][1]
    # print(f"Chosen n:{i},{j}")
    # mean1 = (initial[0]+initial[1])/2
    # mean2 = (final[0]+final[1])/2
    # mean = (mean1+mean2)/2
    # std_deviation = 1
    # val1 = math.exp(-(i)**2/(2*std_deviation))
    # val2 = math.exp(-(j)**2/(2*std_deviation))
    # p1 = 1/2*math.pi*val1
    # p2 = 1/2*math.pi*val2
    # print(f"Normalising : {int(val1)},{int(val2)}")
    # return [int(p1),int(p2)]





def edge_points(pt1, pt2, canvas):
    canvas1= np.zeros((200, 600, 3)) 
    points=[]
    cv2.line(canvas1, tuple(pt1), tuple(pt2), (255,0,0), 3)
    for i in range(canvas1.shape[0]):
        for j in range(canvas1.shape[1]):
            if canvas1[i,j,0]==255 :
                points.append([i,j])

    for point in points:
        if Obstacle(point,canvas) :
            return False
    return True




# def get_integer_line_points(pt1, pt2):
#     x1, y1 = pt1
#     x2, y2 = pt2
#     dx = abs(x2 - x1)
#     dy = abs(y2 - y1)
#     sx = -1 if x1 > x2 else 1
#     sy = -1 if y1 > y2 else 1
#     err = dx - dy
#     line_points = []
#     while True:
#         line_points.append((int(x1), int(y1)))
#         if x1 == x2 and y1 == y2:
#             break
#         e2 = 2 * err
#         if e2 > -dy:
#             err -= dy
#             x1 += sx
#         if e2 < dx:
#             err += dx
#             y1 += sy
#     return line_points




def rewire(neighbor, node, node_dict, canvas, initial, final) :
        parent=[]
#     if collision_free_edge(node, neighbor, canvas) :
        cost_present = Euclidean_dist(initial, node) + Euclidean_dist(neighbor, node) + Euclidean_dist(node, final)
        cost_previous = node_dict[tuple(node)][1]
        if cost_present < cost_previous :
            node_dict[tuple(node)] = [neighbor, cost_present]
            parent = neighbor
        if len(parent)!=0 :
            return parent
        else :
            return node_dict[tuple(node)][0]
        
        
def Nearest(T, node):
    nearest = T[0]
    for i in range(len(T)) :
        if Euclidean_dist(node, T[i]) < Euclidean_dist(nearest , node) :
            nearest = T[i]
    return nearest



def Sample(n,canvas):
    i = random.randint(0, canvas.shape[0]-1)
    j = random.randint(0, canvas.shape[1]-1)
    return [i,j]

def backtrack(initial_state, final_state, edges, canvas):
    
    state = initial_state.copy()
    path = []
    while True:
        node = edges[tuple(state)]
        path.append(state)
        # cv2.line(canvas, tuple(state), tuple(node), (0, 255, 230), 3)
        if(tuple(node) == tuple(final_state)):
            path.append(final_state)
            print("Back Tracking Done!")
            break
        state = node.copy()
    return path


def Euclidean_dist(node1, node2) :
    return math.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)
        
    
def generate_neighbors(node,T, radius) :
    neighbors= []
    for i in range(len(T)) :
        if Euclidean_dist(node, T[i]) <= radius:
            neighbors.append(T[i])
    return neighbors

def Obstacle(node,canvas) :
    if node[0]< 200 and node[1]<600 and canvas[int(round(node[0]))][int(round(node[1]))][0]==255 and canvas[int(round(node[0]))][int(round(node[1]))][1]==255 and canvas[int(round(node[0]))][int(round(node[1]))][2]==255 :
        return False
    else :
        return True
  #  print (node)

def Obstacle2(node,canvas2) :
    if node[0]< 200 and node[1]<600 and canvas2[int(round(node[0]))][int(round(node[1]))][0]==255 and canvas2[int(round(node[0]))][int(round(node[1]))][1]==255 and canvas2[int(round(node[0]))][int(round(node[1]))][2]==255 :
        return False
    else :
        return True
  #  print (node)
        
def Steer(Znearest, Zrand, L) :
    point=[]
    #print("Steer")
    Znearest = np.array(Znearest)
    Zrand = np.array(Zrand)
    if Euclidean_dist(Znearest, Zrand) > L :
        dist = Euclidean_dist(Znearest , Zrand)
        #unit_vector = [-(-Zrand[0] + Znearest[0])/dist , -(Zrand[1] - Znearest[1])/dist]
        delta_z = Zrand - Znearest
    
    # Compute the distance from z_nearest to z_rand
        d = np.linalg.norm(delta_z)
    
    # Compute the unit vector from z_nearest towards z_rand
        if d == 0:
        # Handle the case where z_nearest and z_rand are the same point
            unit_vector = np.zeros_like(delta_z)
        else:
            unit_vector = delta_z / d
    
        point = [int(round(Znearest[0] + L*unit_vector[0])) ,int(round( Znearest[1] + L*unit_vector[1]))]
#         print( point)
        print("Inside steer")
        print(point)
        if round(point[0]) < canvas.shape[0] and point[0]> 0 and round(point[1]) < canvas.shape[1] and point[1]>0 :
            return point
        else : 
            return Zrand
        
    else :
        print("Steer")
        print(Zrand)
        return Zrand
    



def Backtrack1(closed_list, T, initial_state, final_state, canvas):
    keys = closed_list.keys()    # Returns all the nodes that are explored
    path_stack = []    # Stack to store the path from start to goal
    
    # Visualizing the explored nodes
    keys = list(keys)
    
    for key in keys:
        p_node = closed_list[tuple(key)]
        cv2.circle(canvas,(int(key[1]),int(key[0])),2,(0,0,255),-1)
        cv2.circle(canvas,(int(p_node[1]),int(p_node[0])),2,(0,0,255),-1)
        canvas = cv2.arrowedLine(canvas, (int(p_node[1]),int(p_node[0])), (int(key[1]),int(key[0])), (0,255,0), 1, tipLength = 0.2)
        cv2.imshow("RRT* Exploration and Optimal Path Visualization",canvas)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        
    print(f"Total no.of nodes explored: {len(T)}")
    parent_node = closed_list[tuple(final_state)]
    path_stack.append(final_state)    # Appending the final state because of the loop starting condition
    
    while(tuple(parent_node) != tuple(initial_state)):
        path_stack.append(parent_node)
        parent_node = closed_list[tuple(parent_node)]
    
    path_stack.append(initial_state)    # Appending the initial state because of the loop breaking condition
    draw_line(path_stack,canvas)
    print("\nOptimal Path: ")
    
    start_node = path_stack.pop()
    print(start_node)
    
    for i in range(len(path_stack)):
        print(path_stack.pop())
   # Visualizing the optimal path
    # for i in range(1,len(path_stack)):
    #     cv2.line(canvas, tuple([path_stack[i-1][1],path_stack[i-1][0]]), tuple([path_stack[i][1],path_stack[i][0]]), (0, 255, 0), 3)

    # cv2.imshow("Path",canvas)
    cv2.waitKey()
    cv2.destroyAllWindows()   
    
def draw_line(path_stack, canvas):
    path_canvas = canvas.copy()
    print("------DRAWING THE PATH------")
    for i in range(1,len(path_stack)):
        cv2.line(path_canvas, tuple([path_stack[i-1][1],path_stack[i-1][0]]), tuple([path_stack[i][1],path_stack[i][0]]), (0, 255, 0), 3)
    cv2.imshow("Path",path_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def goal_thresh(start,goal):
    x1=start[0]
    y1=start[1]
    
    x2=goal[0]
    y2=goal[1]
    #goal_angle=goal[2]
    if((x1-x2)**2+(y1-y2)**2<=10**2 ):
        return True
    else:
        return False

import math
import cv2 
import numpy as np
import random
import copy
import time

if __name__ == '__main__': 
    canvas = np.zeros((200, 600, 3)) 
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]) :
            canvas[i][j][0] = 255
            canvas[i][j][1] = 255
            canvas[i][j][2] = 255
    rad=10.5
    L=13.8
    cle=input("Enter Robot clearance") 
    start=[] 
    goal=[]    
    bloat=int(float(rad)+float(cle))
     
    #rectangle1 bloat
    for j in range(150-bloat,166+bloat):
        for i in range(126+bloat):
            canvas[i][j]=[0,0,255]
            
            
    #rectangle2 bloat
    for j in range(250-bloat,266+bloat):
        for i in range(75-bloat,200):
            canvas[i][j]=[0,0,255]        
            
    #rectangle1
    for j in range(150,166):
        for i in range(126):
            canvas[i][j]=[255,0,0]
            
            
    #rectangle2
    for j in range(250,266):
        for k in range(75,200):
            canvas[k][j]=[255,0,0]
            
            
    #circle bloat
    for i in range(0,200):
        for j in range(0,600):
            if(np.sqrt((i-90)**2 + (j-400)**2 )<=50+bloat):
                canvas[i,j]=[0,0,255]
            
    #circle        
    for i in range(0,200):
        for j in range(0,600):
            if(np.sqrt((i-90)**2 + (j-400)**2 )<=50):
                canvas[i,j]=[255,0,0]

    #bloating edges - y plane
    for i,j in zip(range(0,bloat+1),range(600-bloat,600)):
        for k in range(200):
                   canvas[k,i]=[0,0,255]
                   canvas[k,j]=[0,0,255]
    
    
    #bloating edges - x plane
    for i,j in zip(range(0,bloat+1),range(200-bloat,200)):
        for k in range(600):
                   canvas[i,k]=[0,0,255]
                   canvas[j,k]=[0,0,255]
                   
    while(1):
        x1=input("Enter start point X coordinate: ")
        y1=input("Enter start point Y coordinate: ")      
#         theta1=input("Enter start point orientation (Enter a multiple of 30): ")      
        x2=input("Enter goal point X coordinate: ") 
        y2=input("Enter goal point Y coordinate: ")
#         rpm1=int(input("Enter Left Wheel RPM"))
#         rpm2=int(input("Enter Right Wheel RPM"))
        if(canvas[200-int(y1),int(x1),0]!=255 or canvas[200-int(y1),int(x1),2]!=255):
            print("Start Node in obstacle space, try again")
        elif(canvas[200-int(y2),int(x2),0]!=255 or canvas[200-int(y2),int(x2),2]!=255):
            print("Goal Node in obstacle space, try again")
#         elif(int(theta1)%30!=0):
#             print("Enter angles in multiples of 30")
        elif([x1,y1]==[x2,y2]):
            print("Goal reached..")
           
        else:
            break
           
        
    start.append(200-int(y1)) 
    start.append(int(x1)) 
    #start.append(360-int(theta1))
    goal.append(200-int(y2))
    goal.append(int(x2)) 
    print(f"Start point:{start}, Goal point:{goal}")
    #goal.append(int(theta2))
    cv2.circle(canvas,(start[1],start[0]),2,(0,0,255),-1)
    cv2.circle(canvas,(goal[1],goal[0]),2,(0,200,0),-1)
    cv2.imshow("canvas in main",canvas)
    rrt_star(start , goal , canvas)
    #rrt_star_n(start, goal, canvas)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()