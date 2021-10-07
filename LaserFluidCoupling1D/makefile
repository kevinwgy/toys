.DEFAULT_GOAL := all

HGVERSION:= $(shell hg parents --template 'hgid: {node|short}')

OBJS            = input.o integrate.o output.o main.o
#CFLAGS          = -I${BOOST_INC}
CCPPFLAGS        = -O2 -std=c++14 -fopenmp
CLEANFILE      = LaserFluid1D

all: $(OBJS) parser
	$(CXX) $(OBJS) -o LaserFluid1D -Lparser -lparser -fopenmp
#	${RM} ${OBJS}

main.o: main.cpp input.h output.h hgversion.h .hg/cache/*
	echo "#define hgRevisionNo "\"`hg id -n`\" > hgversion.h
	echo "#define hgRevisionHash "\"`hg id -i`\" >> hgversion.h
	$(CXX) -c main.cpp ${CXX_FLAGS} ${CXXFLAGS} ${CCPPFLAGS}

input.o: input.cpp input.h parser/Assigner.h parser/Dictionary.h
	$(CXX) -c input.cpp  ${CXX_FLAGS} ${CXXFLAGS} ${CCPPFLAGS}

integrate.o: integrate.cpp input.h
	$(CXX) -c integrate.cpp  ${CXX_FLAGS} ${CXXFLAGS} ${CCPPFLAGS}

output.o: output.cpp output.h input.h hgversion.h 
	$(CXX) -c output.cpp ${CXX_FLAGS} ${CXXFLAGS} ${CCPPFLAGS}

parser::
##	(cd parser; make "CXX=$(CXX)")
	+make -C parser "CXX=$(CXX)"

clean:
	rm -f $(OBJS) $(CLEANFILE)
	(cd parser; $(MAKE) clean)
