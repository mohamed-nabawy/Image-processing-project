from skimage import io
import imutils
import os
import cv2
from cv2 import *
import numpy as np
from xlrd import open_workbook
from matplotlib import pyplot as plt
import xlsxwriter


#									 read excel file to get answer places
# papers_true_degrees={}

# wb = open_workbook(r'C:\Users\HP\Desktop\ImageProject\material\train.xlsx' ,'utf-8')

# for s in wb.sheets():
#     for row in range(s.nrows):
#    		papers_true_degrees[str(s.cell(row,0).value)] = int(s.cell(row,1).value)



		# Excel file to write the output
workbook = xlsxwriter.Workbook('test_output.xlsx')
worksheet = workbook.add_worksheet()



	
#os.system(r"python C:\Users\HP\Anaconda2\Lib\site-packages\alyn\deskew.py -i %s\%s -o deskewed%d.png" %(path,'S_2_hppscan15.png',counter))

# 		Some Initializations


path =r'E:\train'
path2=r'F:\test'

row =0 

error = 0
# check if this cell black or white
def black_or_white(st_answer_cell): # right version
	for r in range(len(st_answer_cell)):
				for c in range(len(st_answer_cell[0])):
					if st_answer_cell[r,c] > 203 :   #[row,col]   203
						st_answer_cell[r,c] =0 #white
					else:
						st_answer_cell[r,c] =1

	return (np.sum(st_answer_cell)>(0.22 * st_answer_cell.size)) #0.22

def black_or_white_2(st_answer_cell): # wrong version
	for r in range(len(st_answer_cell)):
				for c in range(len(st_answer_cell[0])):
					if st_answer_cell[r,c] > 160 :   # make two thresholds
						st_answer_cell[r,c] =0 #white
					else:
						st_answer_cell[r,c] =1

	return (np.sum(st_answer_cell)>(0.28 * st_answer_cell.size))

# 		Begin finding places of right answer with respect to 'Q 01' 

answers_places= [2, 3, 1, 1, 4, 1, 3, 3, 1, 3, 1, 2, 3, 3, 2, 1, 4, 2, 3, 2, 4, 3, 4, 2, 4, 3, 4, 4, 2, 3, 2, 2, 4, 3, 2, 3, 2, 3, 3, 1, 2, 2, 3, 3, 2]

other_places_for={1:[2,3,4],2:[1,3,4],3:[1,2,4],4:[1,2,3]}# check other places to see if st marked two cells



method = cv2.TM_SQDIFF_NORMED






#		 		Actual work : looping through the directory 


for file_name in os.listdir(path2):

	image = cv2.imread(r'%s\%s'%(path2 , file_name ) )

	image = cv2.GaussianBlur(image,(5,5),0) # 0 means take sigma x and sigma y from the kernel(5x5) # see pic after filtering
	
	gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # from RGB to gray scale

	gray = cv2.bitwise_not(gray) # inverse colors

	#gray = gray [400:,:800] # get the text part from image

	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # thresholding


	coords = np.column_stack(np.where(thresh > 0))  #  Stack 1-D arrays as columns into a 2-D array.

# 					showing or drawing the image using pyplot			

	
	# ax3.imshow(thresh, interpolation='none')
	# ax3.set_title('larger figure')
	# plt.show()

	angle = cv2.minAreaRect(coords)[-1] # rectangle that wraps the text up  # rect = ((center_x,center_y),(width,height),angle)

	#print angle




	if angle < -45:  # -8X.xy >> clockwise-rotated 	, else	-x.x(small negative) >>	anti-clockwise rotated
		angle = (90 + angle)
	# # otherwise, just take the inverse of the angle to make
	# # it positive

	
	#if file_name=='S_21_hppscan132.png':
	#print file_name ,' ', angle				# -ve is skewed anti-clockwise , +ve  is skewed clockwise 

	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, -angle, 1.0) # -ve rotates the matrix right  , +ve rotates the matrix left(counter clock wise)
	image = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	#rotated = imutils.rotate_bound(image, angle ) # +ve rotates the matrix right  , -ve rotates the matrix left
	#io.imsave(file_name, rotated.astype(np.uint8)) # +ve rotate clockwise

	small_image = cv2.imread('pattern.png')
	large_image = image

	result = cv2.matchTemplate(small_image, large_image, method)

	# We want the minimum squared difference
	mn,_,mnLoc,_ = cv2.minMaxLoc(result)

	# Draw the rectangle:
	# Extract the coordinates of our best match
	MPx,MPy = mnLoc

	# Step 2: Get the size of the template. This is the same size as the match.
	trows,tcols = small_image.shape[:2]
	

	large_image = cv2.cvtColor(large_image, cv2.COLOR_RGB2GRAY)

	Qcounter=0 
	black_counter=0

	MPy -= 1
	MPx +=(tcols+27) # pattern width + white space 

	for answer in answers_places[0:15] :
		Qcounter+=1  #to  print the question number ,,,,Fakess
		temp_MPx = MPx + 42 * (answer-1) # 42 is the the cell width + white space
		st_answer = large_image[MPy:(MPy+40) , temp_MPx : (temp_MPx+24)]# 40 to take all range of y even if the image is alittle inclined
		
		#		Thresholding @ 0.75 for black of 255
		black = black_or_white(st_answer)
		
		
		if black : #if he marked correct one >> check other places 
			black_counter+=1
			for other in other_places_for[answer]:
				temp_MPx = MPx + 42 * (other-1) # 42 is the the cell width + white space
				st_answer = large_image[MPy:(MPy+40) , temp_MPx : (temp_MPx+24)]# 40 to take all range of y even if the image is alittle inclined
				another = black_or_white_2(st_answer)	
				if another:
					black_counter-=1
					break

			
		MPy+=40




	#   2nd Column >> 16:30

	MPx,MPy = mnLoc
	MPy -= 1
	MPx +=(tcols+354) # pattern width + white space 

	for answer in answers_places[15:30] :
		Qcounter+=1
		temp_MPx = MPx + 42 * (answer-1)
		st_answer = large_image[MPy:(MPy+40) , temp_MPx : (temp_MPx+24)]
		#		Thresholding @ 0.75 for black of 255
		black = black_or_white(st_answer)

		if black : #if he marked correct one >> check other places 
				black_counter+=1
				for other in other_places_for[answer]:
					temp_MPx = MPx + 42 * (other-1) # 42 is the the cell width + white space
					st_answer = large_image[MPy:(MPy+40) , temp_MPx : (temp_MPx+24)]# 40 to take all range of y even if the image is alittle inclined
					another = black_or_white_2(st_answer)	
					if another:
						black_counter-=1
						break

		MPy+=40




#   3rd Column >> 30:45


	MPx,MPy = mnLoc
	MPy -= 1
	MPx +=(tcols+685) # pattern width + white space 

	for answer in answers_places[30:45] :
		Qcounter+=1
		temp_MPx = MPx + 42 * (answer-1)
		st_answer = large_image[MPy:(MPy+40) , temp_MPx : (temp_MPx+24)]
		#		Thresholding @ 0.75 for black of 255
		black = black_or_white(st_answer)

	
		if black : #if he marked correct one >> check other places 
					black_counter+=1
					for other in other_places_for[answer]:
						temp_MPx = MPx + 42 * (other-1) # 42 is the the cell width + white space
						st_answer = large_image[MPy:(MPy+40) , temp_MPx : (temp_MPx+24)]# 40 to take all range of y even if the image is alittle inclined
						another = black_or_white_2(st_answer)	
						if another:
							black_counter-=1
							break

		MPy+=40

	# train comparison
	#if papers_true_degrees[file_name] != black_counter:
	#	print  file_name ,'  ','wrong' ,black_counter
	#	error+=1
	
	#write to the file
	worksheet.write(row, 0, file_name)
	worksheet.write(row, 1, black_counter)
	row	+=1

workbook.close()

#print error